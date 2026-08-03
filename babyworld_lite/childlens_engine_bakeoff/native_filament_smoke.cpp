// Minimal native MuJoCo + Filament Metal smoke test.
//
// This source is deliberately standalone so the authoritative fallback can be
// tested without Unity.  It renders an off-screen RGB frame via Filament's
// Metal backend; the full episode runner must only be introduced after this
// rendering ownership check succeeds.

#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/SwapChain.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <backend/DriverEnums.h>
#include <backend/PixelBufferDescriptor.h>
#include <mujoco/mujoco.h>
#include <utils/EntityManager.h>
#include <utils/Panic.h>
#include <math/mat4.h>

#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using filament::Camera;
using filament::Engine;
using filament::IndexBuffer;
using filament::Material;
using filament::RenderableManager;
using filament::Renderer;
using filament::Scene;
using filament::Skybox;
using filament::SwapChain;
using filament::VertexBuffer;
using filament::View;
using filament::Viewport;
using filament::backend::Backend;
using filament::backend::PixelBufferDescriptor;
using utils::Entity;
using utils::EntityManager;

namespace {

struct Vertex {
  float x;
  float y;
  float z;
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

struct TraceSample {
  double time_s;
  struct Pose { std::array<double, 3> position; std::array<double, 9> rotation; };
  Pose torso, head, palm, left_finger, right_finger, left_tip, right_tip, target, camera_mount;
  std::array<double, 5> action;
  std::array<double, 5> qpos, qvel;
  std::array<double, 6> head_cvel, target_cvel;
  int contact_count;
  bool left_target_contact, right_target_contact;
  std::array<double, 6> left_target_wrench, right_target_wrench;
  double left_target_distance = 1.0, right_target_distance = 1.0, target_table_distance = 1.0;
};

constexpr std::array<Vertex, 8> kCubeVertices{{
  {-1,-1,-1, 220,172,110,255}, {1,-1,-1, 220,172,110,255},
  {1,1,-1, 220,172,110,255}, {-1,1,-1, 220,172,110,255},
  {-1,-1,1, 220,172,110,255}, {1,-1,1, 220,172,110,255},
  {1,1,1, 220,172,110,255}, {-1,1,1, 220,172,110,255},
}};
constexpr std::array<std::uint16_t, 36> kCubeIndices{{
  0,1,2,0,2,3, 4,6,5,4,7,6, 0,4,5,0,5,1,
  1,5,6,1,6,2, 2,6,7,2,7,3, 4,0,3,4,3,7,
}};

std::vector<std::uint8_t> ReadFile(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(std::string("could not read material package: ") + path);
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void WritePpm(const char* path, const std::vector<std::uint8_t>& pixels,
              std::uint32_t width, std::uint32_t height) {
  std::ofstream output(path, std::ios::binary);
  output << "P6\n" << width << " " << height << "\n255\n";
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto source_y = height - y - 1;  // Filament readback is bottom-up.
    const auto* row = pixels.data() + source_y * width * 4;
    for (std::uint32_t x = 0; x < width; ++x) {
      output.write(reinterpret_cast<const char*>(row + x * 4), 3);
    }
  }
}

struct PhysicalReceipt {
  double maximum_lift_m = 0.0;
  int first_contact_step = -1;
  int left_contact_steps = 0;
  int right_contact_steps = 0;
  bool released = false;
};

struct EpisodeTrace {
  PhysicalReceipt receipt;
  std::vector<TraceSample> samples;
};

EpisodeTrace RunPhysicalEpisode() {
  // The object is a free body for the full run.  The controller only drives
  // wrist and finger actuators; it never writes the target state or uses an
  // equality constraint, parent, weld, teleport, or assist force.
  static constexpr const char* kMjcf = R"(
<mujoco model="native_physical_grasp">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default><joint damping="1"/><geom friction="10 0.01 0.001" condim="4" solref=".01 1" solimp=".99 .99 .001"/>
    <position kp="650" forcelimited="true" forcerange="-180 180"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".2 .2 .2 1"/>
    <body name="table" pos=".45 0 .30"><geom name="table_top" type="box" size=".35 .30 .03" solref="-10000 -100" solimp=".999 .999 .00001" rgba=".45 .25 .12 1"/></body>
    <body name="child_root" pos="0 0 .33">
      <geom name="torso" type="box" pos="0 0 .15" size=".12 .10 .15" rgba=".85 .56 .38 1"/>
      <body name="head" pos="0 0 .36"><joint name="neck_yaw" type="hinge" axis="0 0 1" range="-.7 .7" limited="true"/>
        <geom name="head_geom" type="box" size=".13 .12 .13" mass=".5" rgba=".95 .72 .55 1"/><site name="camera_mount" pos=".105 0 .02"/></body>
      <body name="wrist" pos="0 0 .10"><joint name="wrist_x" type="slide" axis="1 0 0" range="-.08 .55"/>
        <joint name="wrist_z" type="slide" axis="0 0 1" range="-.05 .38"/>
        <geom name="palm" type="box" pos="-.10 0 .06" size=".055 .105 .045" rgba=".93 .61 .43 1"/>
        <body name="left_finger" pos=".06 -.10 -.06"><joint name="left_close" type="slide" axis="0 1 0" range="0 .075"/>
          <geom name="left_finger_geom" type="box" pos=".055 0 0" size=".055 .017 .017" rgba=".93 .61 .43 1"/>
          <geom name="left_tip_geom" type="box" pos=".055 0 -.045" size=".055 .017 .010" rgba=".93 .61 .43 1"/></body>
        <body name="right_finger" pos=".06 .10 -.06"><joint name="right_close" type="slide" axis="0 -1 0" range="0 .075"/>
          <geom name="right_finger_geom" type="box" pos=".055 0 0" size=".055 .017 .017" rgba=".93 .61 .43 1"/>
          <geom name="right_tip_geom" type="box" pos=".055 0 -.045" size=".055 .017 .010" rgba=".93 .61 .43 1"/></body>
      </body>
    </body>
    <body name="target" pos=".47 0 .374"><joint name="target_free" type="free"/>
      <geom name="target_geom" type="box" size=".043 .043 .043" mass=".03" solref="-10000 -100" solimp=".999 .999 .00001" rgba=".97 .78 .10 1"/></body>
  </worldbody>
  <actuator><position name="neck" joint="neck_yaw" kp="60" forcelimited="true" forcerange="-30 30"/><position name="reach" joint="wrist_x" kp="5000" forcelimited="true" forcerange="-1000 1000"/>
    <position name="lift" joint="wrist_z" kp="3000" forcelimited="true" forcerange="-800 800"/><position name="left" joint="left_close" kp="3000" forcelimited="true" forcerange="-500 500"/>
    <position name="right" joint="right_close" kp="3000" forcelimited="true" forcerange="-500 500"/></actuator>
</mujoco>)";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "episode.xml", kMjcf, std::strlen(kMjcf));
  char error[1024] = {};
  mjModel* model = mj_loadXML("episode.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);
  if (!model) throw std::runtime_error(error);
  mjData* data = mj_makeData(model);
  const int target_body = mj_name2id(model, mjOBJ_BODY, "target");
  const int head_body = mj_name2id(model, mjOBJ_BODY, "head");
  const int neck_joint = mj_name2id(model, mjOBJ_JOINT, "neck_yaw");
  const int left_geom = mj_name2id(model, mjOBJ_GEOM, "left_finger_geom");
  const int right_geom = mj_name2id(model, mjOBJ_GEOM, "right_finger_geom");
  const int left_tip_geom = mj_name2id(model, mjOBJ_GEOM, "left_tip_geom");
  const int right_tip_geom = mj_name2id(model, mjOBJ_GEOM, "right_tip_geom");
  const int target_geom = mj_name2id(model, mjOBJ_GEOM, "target_geom");
  const int table_geom = mj_name2id(model, mjOBJ_GEOM, "table_top");
  const int camera_site = mj_name2id(model, mjOBJ_SITE, "camera_mount");
  const int torso_geom = mj_name2id(model, mjOBJ_GEOM, "torso");
  const int head_geom = mj_name2id(model, mjOBJ_GEOM, "head_geom");
  const int palm_geom = mj_name2id(model, mjOBJ_GEOM, "palm");
  EpisodeTrace trace;
  auto& receipt = trace.receipt;
  data->qpos[model->jnt_qposadr[neck_joint]] = -0.42;
  mj_forward(model, data);
  double initial_z = 0.0;
  for (int step = 0; step < 4000; ++step) {
    const double t = step * model->opt.timestep;
    const double reorient = std::clamp((t - 1.7) / 1.5, 0.0, 1.0);
    data->ctrl[0] = t < 1.7 ? -0.52 : (-0.52 + 0.78 * reorient + (t >= 6.0 && t < 11.5 ? .07 * std::sin((t - 6.0) * 1.4) : 0.0));
    data->ctrl[1] = t < 2.5 ? 0.02 : (t < 6.0 ? 0.38 : (t < 11.5 ? 0.39 + .025 * std::sin((t - 6.0) * 5.0) : 0.12));
    const double lift_ramp = std::clamp((t - 5.8) / 1.2, 0.0, 1.0);
    const double place_ramp = std::clamp((t - 9.5) / 2.0, 0.0, 1.0);
    const double held_lift = lift_ramp * (.27 + .012 * std::sin((t - 6.0) * 4.0));
    data->ctrl[2] = t < 5.8 ? 0.00 : (t < 9.5 ? held_lift : (t < 12.2 ? (.03 + (1.0 - place_ramp) * .24) : 0.00));
    data->ctrl[3] = (t >= 4.0 && t < 11.5) ? 0.050 : 0.0;
    data->ctrl[4] = (t >= 4.0 && t < 11.5) ? 0.050 : 0.0;
    mj_step(model, data);
    const double z = data->xpos[3 * target_body + 2];
    if (step == 750) initial_z = z;
    if (step >= 750) receipt.maximum_lift_m = std::max(receipt.maximum_lift_m, z - initial_z);
    for (int c = 0; c < data->ncon; ++c) {
      const auto& contact = data->contact[c];
      const bool left = (contact.geom1 == left_geom || contact.geom2 == left_geom || contact.geom1 == left_tip_geom || contact.geom2 == left_tip_geom) &&
                        (contact.geom1 == target_geom || contact.geom2 == target_geom);
      const bool right = (contact.geom1 == right_geom || contact.geom2 == right_geom || contact.geom1 == right_tip_geom || contact.geom2 == right_tip_geom) &&
                         (contact.geom1 == target_geom || contact.geom2 == target_geom);
      if (left || right) {
        if (receipt.first_contact_step < 0) receipt.first_contact_step = step;
        receipt.left_contact_steps += left;
        receipt.right_contact_steps += right;
      }
    }
    if (step % 8 == 0) {
      TraceSample sample{};
      sample.time_s = data->time;
      auto record_geom = [&](int geom, TraceSample::Pose& pose) {
        std::copy_n(data->geom_xpos + 3 * geom, 3, pose.position.begin());
        std::copy_n(data->geom_xmat + 9 * geom, 9, pose.rotation.begin());
      };
      record_geom(torso_geom, sample.torso); record_geom(head_geom, sample.head); record_geom(palm_geom, sample.palm);
      record_geom(left_geom, sample.left_finger); record_geom(right_geom, sample.right_finger);
      record_geom(left_tip_geom, sample.left_tip); record_geom(right_tip_geom, sample.right_tip);
      record_geom(target_geom, sample.target);
      std::copy_n(data->site_xpos + 3 * camera_site, 3, sample.camera_mount.position.begin());
      std::copy_n(data->site_xmat + 9 * camera_site, 9, sample.camera_mount.rotation.begin());
      std::copy_n(data->ctrl, 5, sample.action.begin());
      const std::array<int, 5> joints{neck_joint,
          mj_name2id(model, mjOBJ_JOINT, "wrist_x"), mj_name2id(model, mjOBJ_JOINT, "wrist_z"),
          mj_name2id(model, mjOBJ_JOINT, "left_close"), mj_name2id(model, mjOBJ_JOINT, "right_close")};
      for (size_t i = 0; i < joints.size(); ++i) {
        sample.qpos[i] = data->qpos[model->jnt_qposadr[joints[i]]];
        sample.qvel[i] = data->qvel[model->jnt_dofadr[joints[i]]];
      }
      std::copy_n(data->cvel + 6 * head_body, 6, sample.head_cvel.begin());
      std::copy_n(data->cvel + 6 * target_body, 6, sample.target_cvel.begin());
      sample.contact_count = data->ncon;
      for (int c = 0; c < data->ncon; ++c) {
        const auto& contact = data->contact[c];
        const bool left = (contact.geom1 == left_geom || contact.geom2 == left_geom || contact.geom1 == left_tip_geom || contact.geom2 == left_tip_geom) &&
                          (contact.geom1 == target_geom || contact.geom2 == target_geom);
        const bool right = (contact.geom1 == right_geom || contact.geom2 == right_geom || contact.geom1 == right_tip_geom || contact.geom2 == right_tip_geom) &&
                           (contact.geom1 == target_geom || contact.geom2 == target_geom);
        const bool table = (contact.geom1 == table_geom || contact.geom2 == table_geom) &&
                           (contact.geom1 == target_geom || contact.geom2 == target_geom);
        if (left || right) {
          double force[6] = {}; mj_contactForce(model, data, c, force);
          if (left) { sample.left_target_contact = true; sample.left_target_distance = std::min(sample.left_target_distance, contact.dist); std::copy_n(force, 6, sample.left_target_wrench.begin()); }
          if (right) { sample.right_target_contact = true; sample.right_target_distance = std::min(sample.right_target_distance, contact.dist); std::copy_n(force, 6, sample.right_target_wrench.begin()); }
        }
        if (table) sample.target_table_distance = std::min(sample.target_table_distance, contact.dist);
      }
      trace.samples.push_back(sample);
    }
  }
  receipt.released = data->xpos[3 * target_body + 2] < initial_z + 0.02;
  mj_deleteData(data);
  mj_deleteModel(model);
  return trace;
}

bool SameTrace(const EpisodeTrace& left, const EpisodeTrace& right, double* maximum_error) {
  *maximum_error = 0.0;
  if (left.samples.size() != right.samples.size()) return false;
  for (size_t i = 0; i < left.samples.size(); ++i) {
    const auto& a = left.samples[i]; const auto& b = right.samples[i];
    if (std::memcmp(&a, &b, sizeof(TraceSample)) != 0) return false;
  }
  return *maximum_error == 0.0;
}

void WriteTruthStreams(const std::filesystem::path& root, const EpisodeTrace& trace, double replay_error) {
  std::ofstream stream(root / "synchronized_trace.csv");
  stream << "tick,time_s,action_neck,action_reach,action_lift,action_left,action_right,neck_q,wrist_x,wrist_z,left_q,right_q,neck_qvel,wrist_xvel,wrist_zvel,left_qvel,right_qvel,"
         << "head_angx,head_angy,head_angz,head_linx,head_liny,head_linz,target_x,target_y,target_z,target_r00,target_r01,target_r02,target_r10,target_r11,target_r12,target_r20,target_r21,target_r22,"
         << "camera_x,camera_y,camera_z,camera_r00,camera_r01,camera_r02,camera_r10,camera_r11,camera_r12,camera_r20,camera_r21,camera_r22,"
         << "left_contact,right_contact,left_distance,right_distance,target_table_distance,left_normal_force,right_normal_force,object_id\n";
  for (size_t i = 0; i < trace.samples.size(); ++i) {
    const auto& s = trace.samples[i];
    stream << i << ',' << s.time_s;
    for (double value : s.action) stream << ',' << value;
    for (double value : s.qpos) stream << ',' << value;
    for (double value : s.qvel) stream << ',' << value;
    for (double value : s.head_cvel) stream << ',' << value;
    for (double value : s.target.position) stream << ',' << value;
    for (double value : s.target.rotation) stream << ',' << value;
    for (double value : s.camera_mount.position) stream << ',' << value;
    for (double value : s.camera_mount.rotation) stream << ',' << value;
    stream << ',' << s.left_target_contact << ',' << s.right_target_contact << ',' << s.left_target_distance << ',' << s.right_target_distance << ',' << s.target_table_distance
           << ',' << s.left_target_wrench[0] << ',' << s.right_target_wrench[0] << ",target_block_001\n";
  }
  std::ofstream manifest(root / "stream_manifest.json");
  manifest << "{\n  \"clock\": {\"physics_hz\": 250, \"sample_hz\": 31.25, \"duration_s\": 16.0, \"samples\": " << trace.samples.size() << "},\n"
           << "  \"streams\": [\"rgb\", \"depth\", \"object_id\", \"action\", \"proprioception\", \"contact_touch\", \"imu_like\", \"object_state\", \"camera_state\"],\n"
           << "  \"object_identity\": \"target_block_001\",\n  \"same_machine_replay_maximum_error\": " << replay_error << ",\n"
           << "  \"forbidden_attachment_or_assist\": false\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  constexpr std::uint32_t kWidth = 640;
  constexpr std::uint32_t kHeight = 480;
  const auto trace = RunPhysicalEpisode();
  const auto replay = RunPhysicalEpisode();
  double replay_error = 0.0;
  if (!SameTrace(trace, replay, &replay_error)) {
    std::cerr << "same-machine replay mismatch; max_error=" << replay_error << "\n";
    return 8;
  }
  const auto& physical = trace.receipt;
  std::cerr << "physical_lift=" << physical.maximum_lift_m
            << " first_contact_step=" << physical.first_contact_step
            << " left_contacts=" << physical.left_contact_steps
            << " right_contacts=" << physical.right_contact_steps
            << " released=" << physical.released
            << " trace_samples=" << trace.samples.size() << "\n";
  if (trace.samples.size() != 500 || physical.first_contact_step < 0 ||
      physical.left_contact_steps == 0 || physical.right_contact_steps == 0 ||
      physical.maximum_lift_m < 0.04 || !physical.released) {
    return 5;
  }
  const std::filesystem::path output_root(argv[2]);
  std::filesystem::create_directories(output_root / "rgb");
  std::filesystem::create_directories(output_root / "depth");
  std::filesystem::create_directories(output_root / "object_id");
  WriteTruthStreams(output_root, trace, replay_error);
  const auto& camera_probe = trace.samples[125];
  std::cerr << "camera_probe=" << camera_probe.camera_mount.position[0] << ","
            << camera_probe.camera_mount.position[1] << "," << camera_probe.camera_mount.position[2]
            << " target_probe=" << camera_probe.target.position[0] << ","
            << camera_probe.target.position[1] << "," << camera_probe.target.position[2] << "\n";
  std::cerr << "camera_rotation=";
  for (double element : camera_probe.camera_mount.rotation) std::cerr << element << ",";
  std::cerr << "\n";
  const auto material_package = ReadFile(argv[1]);
  Engine* engine = Engine::create(Backend::METAL);
  if (engine == nullptr) {
    return 3;
  }
  auto* swap_chain = engine->createSwapChain(kWidth, kHeight);
  std::cerr << "stage=swap_chain\n";
  auto* renderer = engine->createRenderer();
  std::cerr << "stage=renderer\n";
  auto* scene = engine->createScene();
  std::cerr << "stage=scene_object\n";
  auto* view = engine->createView();
  std::cerr << "stage=view_object\n";
  view->setScene(scene);
  std::cerr << "stage=view_scene\n";
  view->setViewport({0, 0, kWidth, kHeight});
  std::cerr << "stage=viewport\n";
  Renderer::ClearOptions clear_options;
  clear_options.clear = true;
  clear_options.clearColor = {0.11, 0.16, 0.23, 1.0};
  renderer->setClearOptions(clear_options);
  std::cerr << "stage=clear_options\n";

  Entity camera_entity = EntityManager::get().create();
  Camera* camera = engine->createCamera(camera_entity);
  std::cerr << "stage=camera\n";
  camera->setProjection(68.0, static_cast<double>(kWidth) / kHeight, 0.03, 20.0,
                        Camera::Fov::VERTICAL);
  std::cerr << "stage=projection\n";
  camera->lookAt({0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
  std::cerr << "stage=look_at\n";
  view->setCamera(camera);
  std::cerr << "stage=view_camera\n";

  filament::Material* material = nullptr;
  try {
    material = Material::Builder().package(material_package.data(), material_package.size()).build(*engine);
  } catch (const utils::PostconditionPanic& error) {
    std::cerr << "Filament material postcondition: " << error.what() << "\n";
    return 7;
  }
  std::cerr << "stage=material\n";
  auto* vertices = VertexBuffer::Builder()
      .vertexCount(kCubeVertices.size()).bufferCount(1)
      .attribute(filament::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, sizeof(Vertex))
      .attribute(filament::COLOR, 0, VertexBuffer::AttributeType::UBYTE4, 12, sizeof(Vertex))
      .normalized(filament::COLOR)
      .build(*engine);
  vertices->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(kCubeVertices.data(), sizeof(kCubeVertices), nullptr));
  auto* indices = IndexBuffer::Builder().indexCount(kCubeIndices.size()).bufferType(IndexBuffer::IndexType::USHORT).build(*engine);
  indices->setBuffer(*engine, IndexBuffer::BufferDescriptor(kCubeIndices.data(), sizeof(kCubeIndices), nullptr));
  auto& transforms = engine->getTransformManager();
  std::vector<filament::MaterialInstance*> material_instances;
  auto make_cube = [&](filament::math::float4 color) {
    Entity entity = EntityManager::get().create();
    transforms.create(entity);
    auto* instance = material->createInstance();
    instance->setParameter("baseColor", color);
    material_instances.push_back(instance);
    RenderableManager::Builder(1).boundingBox({{-1,-1,-1},{1,1,1}})
        .material(0, instance)
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vertices, indices, 0, kCubeIndices.size())
        .culling(false).build(*engine, entity);
    scene->addEntity(entity);
    return entity;
  };
  Entity floor = make_cube({.30f,.45f,.63f,1}), table = make_cube({.48f,.23f,.08f,1});
  Entity shelf = make_cube({.55f,.42f,.25f,1}), wall = make_cube({.72f,.62f,.50f,1});
  Entity book = make_cube({.18f,.56f,.52f,1}), blocks = make_cube({.72f,.18f,.18f,1});
  Entity torso = make_cube({.83f,.43f,.25f,1});
  Entity head = make_cube({.96f,.70f,.52f,1});
  Entity palm = make_cube({.94f,.63f,.43f,1}), left_finger = make_cube({.94f,.63f,.43f,1});
  Entity right_finger = make_cube({.94f,.63f,.43f,1}), left_tip = make_cube({.94f,.63f,.43f,1});
  Entity right_tip = make_cube({.94f,.63f,.43f,1}), target = make_cube({.96f,.74f,.06f,1});
  const std::vector<filament::math::float4> rgb_colors{
      {.30f,.45f,.63f,1}, {.48f,.23f,.08f,1}, {.55f,.42f,.25f,1}, {.72f,.62f,.50f,1},
      {.18f,.56f,.52f,1}, {.72f,.18f,.18f,1}, {.83f,.43f,.25f,1}, {.96f,.70f,.52f,1},
      {.94f,.63f,.43f,1}, {.94f,.63f,.43f,1}, {.94f,.63f,.43f,1}, {.94f,.63f,.43f,1},
      {.94f,.63f,.43f,1}, {.96f,.74f,.06f,1}};
  const std::vector<filament::math::float4> id_colors{
      {.04f,.04f,.04f,1}, {.10f,.10f,.10f,1}, {.16f,.16f,.16f,1}, {.22f,.22f,.22f,1},
      {.28f,.28f,.28f,1}, {.34f,.34f,.34f,1}, {.40f,.40f,.40f,1}, {.46f,.46f,.46f,1},
      {.52f,.52f,.52f,1}, {.58f,.58f,.58f,1}, {.64f,.64f,.64f,1}, {.70f,.70f,.70f,1},
      {.76f,.76f,.76f,1}, {.95f,.95f,.95f,1}};
  using filament::math::float3;
  using filament::math::mat4f;
  auto set_cube = [&](Entity entity, float3 position, float3 scale) {
    transforms.setTransform(transforms.getInstance(entity), mat4f::translation(position) * mat4f::scaling(scale));
  };
  auto set_pose = [&](Entity entity, const TraceSample::Pose& pose, float3 scale) {
    const auto& r = pose.rotation; const auto& p = pose.position;
    const mat4f world(filament::math::float4{static_cast<float>(r[0]), static_cast<float>(r[3]), static_cast<float>(r[6]), 0},
                      filament::math::float4{static_cast<float>(r[1]), static_cast<float>(r[4]), static_cast<float>(r[7]), 0},
                      filament::math::float4{static_cast<float>(r[2]), static_cast<float>(r[5]), static_cast<float>(r[8]), 0},
                      filament::math::float4{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]), 1});
    transforms.setTransform(transforms.getInstance(entity), world * mat4f::scaling(scale));
  };
  set_cube(floor, {0.45f, 0.0f, 0.0f}, {1.4f, 1.2f, 0.03f});
  set_cube(table, {0.45f, 0.0f, 0.30f}, {0.35f, 0.30f, 0.03f});
  set_cube(shelf, {0.95f, -0.28f, 0.45f}, {0.12f, 0.08f, 0.42f});
  set_cube(wall, {1.42f, 0.0f, 0.75f}, {0.03f, 1.2f, 0.75f});
  set_cube(book, {.78f, .14f, .43f}, {.08f,.05f,.10f});
  set_cube(blocks, {.92f, -.12f, .40f}, {.07f,.07f,.07f});
  set_cube(torso, {0.0f, 0.0f, 0.45f}, {0.12f, 0.10f, 0.30f});
  std::cerr << "stage=scene\n";
  scene->setSkybox(Skybox::Builder().color({0.11f, 0.16f, 0.23f, 1.0f}).build(*engine));

  for (size_t frame = 0; frame < trace.samples.size(); ++frame) {
    const auto& sample = trace.samples[frame];
    // Every visible dynamic mesh consumes its corresponding MuJoCo geom pose.
    set_pose(torso, sample.torso, {.12f,.10f,.15f}); set_pose(head, sample.head, {.13f,.12f,.13f});
    set_pose(palm, sample.palm, {.055f,.105f,.045f});
    set_pose(left_finger, sample.left_finger, {.055f,.017f,.017f});
    set_pose(right_finger, sample.right_finger, {.055f,.017f,.017f});
    set_pose(left_tip, sample.left_tip, {.055f,.017f,.010f}); set_pose(right_tip, sample.right_tip, {.055f,.017f,.010f});
    set_pose(target, sample.target, {.043f,.043f,.043f});
    const auto& r = sample.camera_mount.rotation; const auto& p = sample.camera_mount.position;
    // Full mount frame plus one fixed optical transform: the optical axis is
    // child-head +X with a 45-degree downward pitch, and optical +Y is head
    // +Z.  The constants are fixed calibration, not a target-facing path.
    auto local_to_world = [&](double x, double y, double z) {
      return float3{static_cast<float>(x * r[0] + y * r[1] + z * r[2]),
                    static_cast<float>(x * r[3] + y * r[4] + z * r[5]),
                    static_cast<float>(x * r[6] + y * r[7] + z * r[8])};
    };
    const float3 camera_eye = float3{static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])}
        + local_to_world(.08, 0.0, .01);
    const float3 forward = local_to_world(.707, 0.0, -.707);
    const float3 up = local_to_world(.707, 0.0, .707);
    if (frame == 0) std::cerr << "stage=trace_pose\n";
    camera->lookAt(camera_eye, camera_eye + forward, up);
    if (frame == 0) std::cerr << "stage=trace_camera\n";
    const auto render_channel = [&](const char* channel, float render_mode,
                                    const std::vector<filament::math::float4>& colors) {
      for (size_t entity = 0; entity < material_instances.size(); ++entity) {
        material_instances[entity]->setParameter("baseColor", colors[entity]);
        material_instances[entity]->setParameter("cameraEye", filament::math::float4{camera_eye, 1.0f});
        material_instances[entity]->setParameter("cameraForward", filament::math::float4{forward, 0.0f});
        material_instances[entity]->setParameter("renderMode", render_mode);
      }
      if (!renderer->beginFrame(swap_chain)) throw std::runtime_error("Filament skipped an offscreen frame");
      renderer->render(view);
      std::vector<std::uint8_t> pixels(kWidth * kHeight * 4);
      renderer->readPixels(0, 0, kWidth, kHeight,
          PixelBufferDescriptor(pixels.data(), pixels.size(),
              PixelBufferDescriptor::PixelDataFormat::RGBA,
              PixelBufferDescriptor::PixelDataType::UBYTE));
      renderer->endFrame(); engine->flushAndWait();
      std::ostringstream name;
      name << output_root.string() << "/" << channel << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".ppm";
      WritePpm(name.str().c_str(), pixels, kWidth, kHeight);
    };
    render_channel("rgb", 0.0f, rgb_colors);
    render_channel("object_id", 1.0f, id_colors);
    render_channel("depth", 2.0f, rgb_colors);
  }
  std::cerr << "stage=frames_complete\n";
  for (Entity entity : {floor, table, shelf, wall, book, blocks, torso, head, palm, left_finger, right_finger, left_tip, right_tip, target}) engine->destroy(entity);
  std::cerr << "stage=entities_destroyed\n";
  for (auto* instance : material_instances) engine->destroy(instance);
  engine->flushAndWait();
  std::cerr << "stage=instances_destroyed\n";
  engine->destroy(material);
  std::cerr << "stage=material_destroyed\n";
  engine->destroy(vertices);
  engine->destroy(indices);
  std::cerr << "stage=geometry_destroyed\n";
  engine->destroyCameraComponent(camera_entity);
  EntityManager::get().destroy(camera_entity);
  engine->destroy(view);
  engine->destroy(scene);
  engine->destroy(renderer);
  engine->destroy(swap_chain);
  std::cerr << "stage=engine_destroy\n";
  Engine::destroy(&engine);
  std::cerr << "stage=done\n";
  return 0;
}
