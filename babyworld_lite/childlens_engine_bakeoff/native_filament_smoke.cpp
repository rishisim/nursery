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
#include <filament/RenderableManager.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/SwapChain.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <backend/DriverEnums.h>
#include <backend/PixelBufferDescriptor.h>
#include <utils/EntityManager.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
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

constexpr std::array<Vertex, 3> kVertices{{
    {-0.75f, -0.62f, 0.0f, 244, 179, 82, 255},
    {0.75f, -0.62f, 0.0f, 108, 196, 159, 255},
    {0.0f, 0.78f, 0.0f, 125, 148, 227, 255},
}};
constexpr std::array<std::uint16_t, 3> kIndices{{0, 1, 2}};

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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  constexpr std::uint32_t kWidth = 320;
  constexpr std::uint32_t kHeight = 240;
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
  camera->setProjection(Camera::Projection::ORTHO, -1.0, 1.0, -0.75, 0.75, 0.1, 10.0);
  std::cerr << "stage=projection\n";
  camera->lookAt({0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
  std::cerr << "stage=look_at\n";
  view->setCamera(camera);
  std::cerr << "stage=view_camera\n";

  auto* material = Material::Builder().package(material_package.data(), material_package.size()).build(*engine);
  std::cerr << "stage=material\n";
  auto* vertices = VertexBuffer::Builder()
      .vertexCount(kVertices.size()).bufferCount(1)
      .attribute(filament::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, sizeof(Vertex))
      .attribute(filament::COLOR, 0, VertexBuffer::AttributeType::UBYTE4, 12, sizeof(Vertex))
      .normalized(filament::COLOR)
      .build(*engine);
  vertices->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(kVertices.data(), sizeof(kVertices), nullptr));
  auto* indices = IndexBuffer::Builder().indexCount(kIndices.size()).bufferType(IndexBuffer::IndexType::USHORT).build(*engine);
  indices->setBuffer(*engine, IndexBuffer::BufferDescriptor(kIndices.data(), sizeof(kIndices), nullptr));
  Entity triangle = EntityManager::get().create();
  RenderableManager::Builder(1)
      .boundingBox({{-1.0f, -1.0f, -0.1f}, {1.0f, 1.0f, 0.1f}})
      .material(0, material->getDefaultInstance())
      .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vertices, indices, 0, kIndices.size())
      .culling(false).build(*engine, triangle);
  scene->addEntity(triangle);
  std::cerr << "stage=scene\n";
  scene->setSkybox(Skybox::Builder().color({0.11f, 0.16f, 0.23f, 1.0f}).build(*engine));

  std::vector<std::uint8_t> pixels(kWidth * kHeight * 4);
  if (!renderer->beginFrame(swap_chain)) {
    return 4;
  }
  std::cerr << "stage=begin_frame\n";
  renderer->render(view);
  std::cerr << "stage=render\n";
  renderer->readPixels(0, 0, kWidth, kHeight,
      PixelBufferDescriptor(pixels.data(), pixels.size(),
          PixelBufferDescriptor::PixelDataFormat::RGBA,
          PixelBufferDescriptor::PixelDataType::UBYTE));
  renderer->endFrame();
  std::cerr << "stage=end_frame\n";
  engine->flushAndWait();
  WritePpm(argv[2], pixels, kWidth, kHeight);
  engine->destroy(triangle);
  engine->destroy(material);
  engine->destroy(vertices);
  engine->destroy(indices);
  engine->destroyCameraComponent(camera_entity);
  EntityManager::get().destroy(camera_entity);
  engine->destroy(view);
  engine->destroy(scene);
  engine->destroy(renderer);
  engine->destroy(swap_chain);
  Engine::destroy(&engine);
  return 0;
}
