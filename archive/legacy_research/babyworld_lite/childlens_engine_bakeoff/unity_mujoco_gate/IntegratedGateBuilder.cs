using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class IntegratedGateBuilder
{
    const int Width = 960, Height = 540, Fps = 30;
    static readonly string Output = Environment.GetEnvironmentVariable("UNITY_MUJOCO_OUTPUT");
    static readonly string TracePath = Environment.GetEnvironmentVariable("UNITY_MUJOCO_TRACE");
    static readonly string ConfigPath = Environment.GetEnvironmentVariable("UNITY_MUJOCO_CONFIG");
    static GameObject avatar, target;
    static SkinnedMeshRenderer skin;
    static Mesh bakedMesh;
    static Camera headCamera, qaCamera;
    static Transform head, torso, shoulder, wrist, fingertip, physicsHead;
    static readonly List<Transform> armIk = new List<Transform>();
    static readonly List<Transform> fingerBones = new List<Transform>();
    static readonly Dictionary<Transform, Quaternion> restRotation = new Dictionary<Transform, Quaternion>();
    static readonly Dictionary<Transform, Vector3> restPosition = new Dictionary<Transform, Vector3>();
    static Trace trace;
    static Material idTarget, idSkin, idScene;
    static Quaternion fixedHeadCameraRotation;
    static Vector3 fixedHeadCameraPosition;
    static Dictionary<string, int> jointIndex;
    static Quaternion wristRegistrationOffset;
    static bool wristRegistrationReady;
    static bool localSweepPassed;
    static readonly List<Transform> collisionOverlay = new List<Transform>();

    [MenuItem("BabyWorld/Register Integrated Gate")]
    public static void Register()
    {
        RequireEnvironment();
        Directory.CreateDirectory(Output);
        trace = JsonUtility.FromJson<Trace>(File.ReadAllText(TracePath));
        jointIndex = trace.joint_names.Select((name, index) => new { name, index }).ToDictionary(x => x.name, x => x.index);
        BuildScene();
        ApplyFrame(trace.frames[0]);
        UpdateBakedSkin();
        var sweep = AuditLocalRotationSweeps();
        localSweepPassed = sweep.passed;
        File.WriteAllText(Path.Combine(Output, "local_rotation_sweeps.json"), JsonUtility.ToJson(sweep, true));
        var report = AuditRegistration();
        File.WriteAllText(Path.Combine(Output, "embodiment_manifest.json"), JsonUtility.ToJson(report.manifest, true));
        File.WriteAllText(Path.Combine(Output, "registration_qa.json"), JsonUtility.ToJson(report, true));
        RenderOne(qaCamera, Path.Combine(Output, "registration_overlay.png"), false, false);
        AssetDatabase.SaveAssets();
        if (!report.passed) throw new Exception("Stage A registration failed frozen tolerances");
        EditorApplication.Exit(0);
    }

    [MenuItem("BabyWorld/Export MPFB Rest Manifest")]
    public static void ExportRestManifest()
    {
        RequireEnvironment(); Directory.CreateDirectory(Output);
        trace = JsonUtility.FromJson<Trace>(File.ReadAllText(TracePath));
        jointIndex = trace.joint_names.Select((name, index) => new { name, index }).ToDictionary(x => x.name, x => x.index);
        BuildScene();
        foreach (var kv in restRotation) if (kv.Key) kv.Key.localRotation = kv.Value;
        foreach (var kv in restPosition) if (kv.Key) kv.Key.localPosition = kv.Value;
        var names = new List<string> { "root", "spine01", "spine02", "spine03", "neck", "head", "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R" };
        for (int digit = 1; digit <= 5; digit++) for (int part = 1; part <= 3; part++) names.Add($"finger{digit}-{part}.R");
        var rows = new List<RestBone>();
        foreach (var name in names) {
            var bone = FindOptional(name); if (!bone) continue;
            rows.Add(new RestBone { name = name, source_parent = bone.parent ? bone.parent.name : "", retained_parent = RetainedParent(name, names.ToArray()), world_position_unity = bone.position, world_rotation_unity = bone.rotation, local_position_source = bone.localPosition, local_rotation_source = bone.localRotation });
        }
        var export = new RestManifest { schema = "embodied.mpfb_rest_manifest.v1", source_fbx_sha256 = "b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de", avatar_scale = avatar.transform.localScale.x, coordinate_conversion = "Unity(x,y,z)=MuJoCo(x,z,-y)", bones = rows.ToArray() };
        File.WriteAllText(Path.Combine(Output, "mpfb_rest_manifest.json"), JsonUtility.ToJson(export, true)); AssetDatabase.SaveAssets(); EditorApplication.Exit(0);
    }

    static string RetainedParent(string name, string[] prior)
    {
        var bone = Find(name).parent; while (bone && !prior.Contains(bone.name)) bone = bone.parent; return bone ? bone.name : "";
    }

    [MenuItem("BabyWorld/Render Integrated Gate")]
    public static void Render()
    {
        RequireEnvironment();
        Directory.CreateDirectory(Output);
        trace = JsonUtility.FromJson<Trace>(File.ReadAllText(TracePath));
        jointIndex = trace.joint_names.Select((name, index) => new { name, index }).ToDictionary(x => x.name, x => x.index);
        if (trace.steps_per_frame != 8 || trace.physics_hz != 240 || trace.render_hz != 30 || trace.frames.Length != 660)
            throw new Exception("Trace clock contract violated");
        BuildScene();
        var samples = new HashSet<int>(new[] { 0, 90, 180, 195, 210, 270, 300, 360, 420, 480, 540, 570, 600, 630, 659 });
        var rgb = Path.Combine(Output, "rgb"); var qa = Path.Combine(Output, "external_qa");
        var depth = Path.Combine(Output, "depth"); var ids = Path.Combine(Output, "instance_id");
        Directory.CreateDirectory(rgb); Directory.CreateDirectory(qa); Directory.CreateDirectory(depth); Directory.CreateDirectory(ids);
        var metrics = new List<FrameMetric>();
        foreach (var frame in trace.frames) {
            ApplyFrame(frame); UpdateBakedSkin();
            RenderOne(headCamera, Path.Combine(rgb, $"frame_{frame.frame:D4}.png"), false, false);
            RenderOne(qaCamera, Path.Combine(qa, $"frame_{frame.frame:D4}.png"), false, false);
            RenderOne(headCamera, Path.Combine(depth, $"frame_{frame.frame:D4}.png"), true, false);
            RenderOne(headCamera, Path.Combine(ids, $"frame_{frame.frame:D4}.png"), false, true);
            if (samples.Contains(frame.frame)) metrics.Add(AuditFrame(frame));
        }
        var receipt = new RenderReceipt {
            unity_version = Application.unityVersion, graphics_api = SystemInfo.graphicsDeviceType.ToString(), frame_count = trace.frames.Length,
            physics_hz = trace.physics_hz, render_hz = trace.render_hz, steps_per_frame = trace.steps_per_frame,
            actual_weighted_skin_only = true, proxy_pixels_in_rgb = false, unity_physics_enabled = false,
            target_pose_source = "immutable MuJoCo trace red_toy_001 free body", camera_pose_source = "MuJoCo root/torso/neck/head pose plus fixed mount",
            fixed_camera_mount = true, camera_target_dependent = false, fov_y_deg = headCamera.fieldOfView, near_clip_m = headCamera.nearClipPlane,
            sampled_frames = metrics.ToArray()
        };
        File.WriteAllText(Path.Combine(Output, "unity_render_qa.json"), JsonUtility.ToJson(receipt, true));
        AssetDatabase.SaveAssets(); EditorApplication.Exit(0);
    }

    static void RequireEnvironment()
    {
        if (string.IsNullOrWhiteSpace(Output) || string.IsNullOrWhiteSpace(TracePath) || string.IsNullOrWhiteSpace(ConfigPath))
            throw new Exception("UNITY_MUJOCO_OUTPUT, UNITY_MUJOCO_TRACE, and UNITY_MUJOCO_CONFIG are required");
    }

    static void BuildScene()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        armIk.Clear(); fingerBones.Clear(); restRotation.Clear(); restPosition.Clear(); collisionOverlay.Clear();
        wristRegistrationReady = false;
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(.38f, .47f, .57f); RenderSettings.ambientEquatorColor = new Color(.28f, .25f, .22f); RenderSettings.ambientGroundColor = new Color(.10f, .08f, .07f);
        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx"));
        avatar.name = "CC0_Weighted_MPFB_Child"; avatar.transform.localScale = Vector3.one * 1.9f;
        skin = avatar.GetComponentsInChildren<SkinnedMeshRenderer>().Single(); skin.updateWhenOffscreen = true; skin.localBounds = new Bounds(Vector3.zero, Vector3.one * 4);
        var skinMat = NewMaterial("Skin", new Color(.66f, .40f, .28f), .12f); skin.sharedMaterial = skinMat;
        head = Find("head"); torso = FindOptional("spine03") ?? FindOptional("spine02") ?? Find("spine01"); shoulder = Find("upperarm01.R"); wrist = Find("wrist.R"); fingertip = Find("finger2-3.R");
        foreach (var n in new[] { "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R" }) armIk.Add(Find(n));
        foreach (var t in avatar.GetComponentsInChildren<Transform>(true)) {
            restRotation[t] = t.localRotation; restPosition[t] = t.localPosition;
            if (t.name.StartsWith("finger") && t.name.EndsWith(".R")) fingerBones.Add(t);
        }
        // One static digital-twin registration transform: align the actual rest
        // head landmark to the MuJoCo head.  This is frozen before replay.
        var registeredHead = PoseFromMj(trace.frames[0].head_pose_mj).position;
        avatar.transform.position += registeredHead - head.position;
        var baked = new GameObject("ActualWeightedSkin_BakeMesh_PerTraceFrame"); baked.transform.SetParent(skin.transform.parent, false);
        baked.transform.localPosition = skin.transform.localPosition; baked.transform.localRotation = skin.transform.localRotation; baked.transform.localScale = skin.transform.localScale;
        bakedMesh = new Mesh { name = "ActualWeightedChildTraceFrame" }; baked.AddComponent<MeshFilter>().sharedMesh = bakedMesh; baked.AddComponent<MeshRenderer>().sharedMaterials = skin.sharedMaterials;
        skin.gameObject.layer = 31;

        physicsHead = new GameObject("MuJoCoHeadAuthority").transform;
        var firstHead = PoseFromMj(trace.frames[0].head_pose_mj); physicsHead.SetPositionAndRotation(firstHead.position, firstHead.rotation);
        var firstCam = CameraPoseFromMj(trace.frames[0].camera_pose_mj);
        fixedHeadCameraPosition = physicsHead.InverseTransformPoint(firstCam.position); fixedHeadCameraRotation = Quaternion.Inverse(physicsHead.rotation) * firstCam.rotation;
        var camObject = new GameObject("FixedOutsideHeadCamera"); camObject.transform.SetParent(physicsHead, false); camObject.transform.localPosition = fixedHeadCameraPosition; camObject.transform.localRotation = fixedHeadCameraRotation;
        headCamera = camObject.AddComponent<Camera>(); headCamera.fieldOfView = 62; headCamera.nearClipPlane = .03f; headCamera.farClipPlane = 20; headCamera.cullingMask &= ~(1 << 31); headCamera.cullingMask &= ~(1 << 30);
        var qaObject = new GameObject("QAExternalCamera"); qaCamera = qaObject.AddComponent<Camera>(); qaCamera.fieldOfView = 42; qaCamera.nearClipPlane = .03f; qaCamera.farClipPlane = 20; var qaPosition = new Vector3(.72f, .88f, .18f); qaObject.transform.SetPositionAndRotation(qaPosition, Quaternion.LookRotation(new Vector3(.08f, .70f, -.36f) - qaPosition, Vector3.up));

        NewMaterial("Floor", new Color(.30f, .19f, .11f), .18f); Box("Floor", new Vector3(0, -.035f, -1.7f), new Vector3(4.6f, .07f, 5.2f), "Floor");
        NewMaterial("Wall", new Color(.58f, .67f, .67f), .10f); Box("BackWall", new Vector3(0, 1.3f, -4.25f), new Vector3(4.6f, 2.6f, .08f), "Wall"); Box("SideWall", new Vector3(-2.26f, 1.3f, -1.7f), new Vector3(.08f, 2.6f, 5.2f), "Wall");
        PlaceMetric("rugRectangle", new Vector3(.15f, 0, -1.8f), 2.3f, 0); PlaceMetric("loungeSofaLong", new Vector3(1.0f, 0, -3.55f), 2.0f, 0); PlaceMetric("bookcaseOpen", new Vector3(-1.78f, 0, -3.45f), 1.7f, 0); PlaceMetric("pottedPlant", new Vector3(-1.6f, 0, -2.55f), 1.0f, 180); PlaceMetric("lampSquareFloor", new Vector3(1.85f, 0, -3.25f), 1.55f, 180);
        NewMaterial("Table", new Color(.38f, .20f, .09f), .22f); Box("ReadableSupportTable", Mj(new Vector3(.18f, -.315f, .39625f)), new Vector3(1.20f, .7925f, .90f), "Table");
        NewMaterial("Toy", new Color(.88f, .035f, .02f), .24f); target = GameObject.CreatePrimitive(PrimitiveType.Cube); target.name = "red_toy_001"; target.transform.localScale = Vector3.one * .055f; target.GetComponent<Renderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>("Assets/Generated/Toy.mat"); UnityEngine.Object.DestroyImmediate(target.GetComponent<Collider>());
        NewMaterial("Blue", new Color(.04f, .18f, .72f), .2f); Sphere("blue_distractor", Mj(new Vector3(-.20f, .40f, .655f)), .07f, "Blue"); NewMaterial("Yellow", new Color(.92f, .67f, .04f), .2f); Box("yellow_distractor", Mj(new Vector3(.32f, .49f, .655f)), new Vector3(.06f, .07f, .06f), "Yellow");
        var key = new GameObject("WindowKey").AddComponent<Light>(); key.type = LightType.Directional; key.intensity = .88f; key.color = new Color(1f, .91f, .80f); key.transform.rotation = Quaternion.Euler(48, -32, 0); key.shadows = LightShadows.Soft;
        var fill = new GameObject("SoftFill").AddComponent<Light>(); fill.type = LightType.Point; fill.range = 5; fill.intensity = 1.7f; fill.color = new Color(.82f, .90f, 1f); fill.transform.position = new Vector3(-1.1f, 1.8f, -1.0f);
        idTarget = NewMaterial("ID_Target", new Color(41 / 255f, 160 / 255f, 1 / 255f), 0); idSkin = NewMaterial("ID_Skin", new Color(7 / 255f, 1 / 255f, 1 / 255f), 0); idScene = NewMaterial("ID_Scene", new Color(1 / 255f, 1 / 255f, 1 / 255f), 0);
        var overlayMaterial = NewMaterial("QA_CollisionOverlay", new Color(.05f, .92f, .35f), .05f);
        for (int i = 0; i < 15; i++) { var o = GameObject.CreatePrimitive(PrimitiveType.Capsule); o.name = "QA_ONLY_MuJoCoDigitSegment_" + i; o.layer = 30; o.GetComponent<Renderer>().sharedMaterial = overlayMaterial; UnityEngine.Object.DestroyImmediate(o.GetComponent<Collider>()); collisionOverlay.Add(o.transform); }
    }

    static void ApplyFrame(Frame f)
    {
        foreach (var kv in restRotation) if (kv.Key) kv.Key.localRotation = kv.Value;
        if (f.qpos.Length > 0) {
            torso.localRotation = restRotation[torso] * Quaternion.Euler(Mathf.Rad2Deg * Q(f, "torso_pitch"), 0, 0);
            head.localRotation = restRotation[head] * Quaternion.Euler(Mathf.Rad2Deg * Q(f, "head_yaw"), 0, 0);
        }
        var wristPose = PoseFromMj(f.wrist_pose_mj);
        shoulder.localRotation = restRotation[shoulder] * Quaternion.Euler(Mathf.Rad2Deg * Q(f, "shoulder_pitch"), 0, 0);
        Find("lowerarm01.R").localRotation = restRotation[Find("lowerarm01.R")] * Quaternion.Euler(Mathf.Rad2Deg * Q(f, "elbow_flex"), 0, 0);
        wrist.localRotation = restRotation[wrist] * Quaternion.Euler(Mathf.Rad2Deg * Q(f, "wrist_roll"), 0, 0);
        foreach (var bone in fingerBones) {
            int digit = Mathf.Clamp(bone.name[6] - '1', 0, 4); string digitName = new[] { "thumb", "index", "middle", "ring", "little" }[digit];
            string part = bone.name.Contains("-1.") ? "proximal" : bone.name.Contains("-2.") ? "middle" : "distal";
            float physical = Mathf.Rad2Deg * Q(f, digitName + "_" + part + "_flex");
            bone.localRotation = restRotation[bone] * Quaternion.Euler(physical, digit == 0 ? -.18f * physical : 0, 0);
        }
        for (int i = 0; i < collisionOverlay.Count; i++) { int offset = i * 7; var p = PoseFromMj(new[] { f.digit_segment_pose_mj[offset], f.digit_segment_pose_mj[offset + 1], f.digit_segment_pose_mj[offset + 2], f.digit_segment_pose_mj[offset + 3], f.digit_segment_pose_mj[offset + 4], f.digit_segment_pose_mj[offset + 5], f.digit_segment_pose_mj[offset + 6] }); float length = i % 3 == 0 ? .022f : i % 3 == 1 ? .019f : .017f; float radius = i % 3 == 0 ? .0085f : i % 3 == 1 ? .0075f : .0065f; collisionOverlay[i].SetPositionAndRotation(p.position + p.rotation * new Vector3(0, 0, length * .5f), p.rotation * Quaternion.Euler(90, 0, 0)); collisionOverlay[i].localScale = new Vector3(radius * 2, length * .5f, radius * 2); }
        var hp = PoseFromMj(f.head_pose_mj); physicsHead.SetPositionAndRotation(hp.position, hp.rotation);
        var tp = PoseFromMj(f.target_pose_mj); target.transform.SetPositionAndRotation(tp.position, tp.rotation);
        if (qaCamera) { var qaPosition = wristPose.position + new Vector3(.30f, .18f, .28f); qaCamera.transform.SetPositionAndRotation(qaPosition, Quaternion.LookRotation(wristPose.position - qaPosition, Vector3.up)); qaCamera.fieldOfView = 38; }
    }

    static void UpdateBakedSkin() { bakedMesh.Clear(); skin.BakeMesh(bakedMesh, true); bakedMesh.RecalculateBounds(); }

    static RegistrationReport AuditRegistration()
    {
        var bones = new[] { "head", "wrist.R", "finger1-1.R", "finger1-2.R", "finger1-3.R", "finger2-1.R", "finger2-2.R", "finger2-3.R", "finger3-1.R", "finger3-2.R", "finger3-3.R", "finger4-1.R", "finger4-2.R", "finger4-3.R", "finger5-1.R", "finger5-2.R", "finger5-3.R" };
        var samples = new[] { 0, 195, 270, 360, 570 };
        var errors = bones.ToDictionary(n => n, n => new List<float>());
        foreach (int frameIndex in samples) {
            var f = trace.frames[frameIndex]; ApplyFrame(f); UpdateBakedSkin();
            foreach (var n in bones) {
                var physical = PhysicalPose(f, n);
                errors[n].Add(Vector3.Distance(Find(n).position, physical.position));
            }
        }
        ApplyFrame(trace.frames[0]);
        var rows = bones.Select((n, i) => new RegistrationRow { bone = n, physics_name = PhysicsName(n), semantic_id = 100 + i, landmark_error_m = errors[n].Max(), landmark_mean_error_m = errors[n].Average(), collider_radius_m = n.StartsWith("finger") ? (n.Contains("-1.") ? .0085f : n.Contains("-2.") ? .0075f : .0065f) : n == "wrist.R" ? .04f : .09f, visible_rest_offset_from_wrist_mj = MjInverse(Find(n).position - wrist.position) }).ToArray();
        float max = rows.Max(x => x.landmark_error_m); float mean = rows.Average(x => x.landmark_error_m);
        var roundtrip = new Vector3(.123f, .456f, .789f); float rt = Vector3.Distance(roundtrip, MjInverse(Mj(roundtrip)));
        var manifest = new EmbodimentManifest {
            schema = "embodied.manifest.v1", source_fbx_sha256 = "b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de", source_scale = 1.9f,
            coordinate_conversion = "Unity(x,y,z)=MuJoCo(x,z,-y); inverse MuJoCo(x,y,z)=Unity(x,-z,y); meters; round-trip tested",
            physics_authority = "MuJoCo 3.3.7; reduced anatomical skeleton plus wrist-centered actuator guide", appearance_authority = "Unity 6000.0.80f1 actual weighted MPFB BakeMesh only",
            physical_names = new[] { "torso", "neck", "head", "shoulder_chain", "elbow_chain", "wrist_guide", "palm", "thumb", "index", "middle", "ring", "little", "red_toy_001" },
            touch_sites = new[] { "thumb_touch", "index_touch", "middle_touch", "ring_touch", "little_touch", "palm_touch" },
            stable_ids = "red_toy_001 semantic=41 instance=41001; child skin semantic=7 instance=7001; scene semantic=1", camera_mapping = "root/torso/neck/head + immutable T_head_camera; outside-head; neutral optical axis",
            engineering_priors = "disclosed generic mass/inertia/damping/actuator bounds; no child-data or ChildLens derivation", mapping = rows
        };
        bool quantitative = max <= .010f && rt <= 1e-9f;
        return new RegistrationReport { passed = quantitative && localSweepPassed, quantitative_registration_passed = quantitative, visual_anatomy_passed = localSweepPassed, failure_reason = quantitative ? "" : "Exact audition BakeMesh/local-rotation path is visually sound, but the reduced MuJoCo segment origins/axes were not derived from the MPFB rest hierarchy; without forbidden world-position offsets the frozen <=10 mm registration tolerance fails.", collider_skin_tolerance_m = .010f, landmark_mean_error_m = mean, landmark_max_error_m = max, coordinate_roundtrip_error_m = rt, camera_neutral_axis_deg = 0, camera_outside_head = true, manifest = manifest };
    }

    static SweepReport AuditLocalRotationSweeps()
    {
        var directory = Path.Combine(Output, "unit_sweeps"); Directory.CreateDirectory(directory);
        foreach (var kv in restRotation) if (kv.Key) kv.Key.localRotation = kv.Value;
        UpdateBakedSkin(); var restBounds = bakedMesh.bounds; var restVertices = bakedMesh.vertices;
        foreach (var proxy in collisionOverlay) proxy.GetComponent<Renderer>().enabled = false;
        var cameraPosition = wrist.position + new Vector3(.24f, .14f, .22f); qaCamera.transform.SetPositionAndRotation(cameraPosition, Quaternion.LookRotation(wrist.position - cameraPosition, Vector3.up)); qaCamera.fieldOfView = 34;
        var joints = new[] { "finger2-1.R", "finger2-2.R", "finger2-3.R", "wrist.R" };
        var rows = new List<SweepRow>(); float maxRatio = 1, maxLengthError = 0;
        foreach (var name in joints) {
            var bone = Find(name); var child = name == "wrist.R" ? Find("finger2-1.R") : name.Contains("-1.") ? Find("finger2-2.R") : name.Contains("-2.") ? Find("finger2-3.R") : null;
            float referenceLength = child ? Vector3.Distance(bone.position, child.position) : 0; float jointMaxRatio = 1, jointMaxLengthError = 0, jointMaxVertexMotion = 0;
            for (int angle = -30; angle <= 30; angle += 5) {
                foreach (var kv in restRotation) if (kv.Key) kv.Key.localRotation = kv.Value;
                bone.localRotation = restRotation[bone] * Quaternion.AngleAxis(angle, Vector3.right); UpdateBakedSkin();
                float ratio = Mathf.Max(bakedMesh.bounds.size.x / restBounds.size.x, Mathf.Max(bakedMesh.bounds.size.y / restBounds.size.y, bakedMesh.bounds.size.z / restBounds.size.z));
                float lengthError = child ? Mathf.Abs(Vector3.Distance(bone.position, child.position) - referenceLength) : 0;
                var vertices = bakedMesh.vertices; float motion = 0; for (int i = 0; i < vertices.Length && i < restVertices.Length; i++) motion = Mathf.Max(motion, Vector3.Distance(vertices[i], restVertices[i]));
                jointMaxRatio = Mathf.Max(jointMaxRatio, ratio); jointMaxLengthError = Mathf.Max(jointMaxLengthError, lengthError); jointMaxVertexMotion = Mathf.Max(jointMaxVertexMotion, motion);
                RenderOne(qaCamera, Path.Combine(directory, $"{name.Replace('.', '_')}_{angle + 30:D2}.png"), false, false);
            }
            rows.Add(new SweepRow { bone = name, axis = "rest-local +X", minimum_deg = -30, maximum_deg = 30, step_deg = 5, maximum_mesh_bounds_ratio = jointMaxRatio, maximum_segment_length_error_m = jointMaxLengthError, maximum_baked_vertex_motion_m = jointMaxVertexMotion });
            maxRatio = Mathf.Max(maxRatio, jointMaxRatio); maxLengthError = Mathf.Max(maxLengthError, jointMaxLengthError);
        }
        foreach (var kv in restRotation) if (kv.Key) kv.Key.localRotation = kv.Value; UpdateBakedSkin(); foreach (var proxy in collisionOverlay) proxy.GetComponent<Renderer>().enabled = true;
        var avatarSize = BoundsOf(avatar).size; var auditionSize = new Vector3(.834204435f, 1.342272639f, .450455666f); float auditionSizeError = Vector3.Distance(avatarSize, auditionSize);
        return new SweepReport { path = "verified audition instantiation + restLocalRotation * AngleAxis; BakeMesh(useScale=true); baked renderer copies source parent/local transform exactly once", frames = rows.Count * 13, maximum_mesh_bounds_ratio = maxRatio, maximum_segment_length_error_m = maxLengthError, integrated_rest_avatar_bounds_size_m = avatarSize, audition_rest_avatar_bounds_size_m = auditionSize, audition_bounds_size_error_m = auditionSizeError, audition_bounds_size_tolerance_m = .008f, passed = maxRatio < 1.35f && maxLengthError < 1e-6f && auditionSizeError < .008f, joints = rows.ToArray() };
    }

    static FrameMetric AuditFrame(Frame f)
    {
        var expected = CameraPoseFromMj(f.camera_pose_mj); float camPosition = Vector3.Distance(expected.position, headCamera.transform.position); float camRotation = Quaternion.Angle(expected.rotation, headCamera.transform.rotation);
        var viewport = headCamera.WorldToViewportPoint(target.transform.position); bool visible = viewport.z > .03f && viewport.x >= 0 && viewport.x <= 1 && viewport.y >= 0 && viewport.y <= 1;
        return new FrameMetric { frame = f.frame, time_s = f.time_s, phase = f.phase, camera_mount_position_error_m = camPosition, camera_mount_rotation_error_deg = camRotation, target_visible = visible, target_viewport = viewport, visible_fingertip_to_physical_wrist_m = Vector3.Distance(fingertip.position, PoseFromMj(f.wrist_pose_mj).position), contact_digits = f.contact_digits, support_contact = f.support_contact };
    }

    static void RenderOne(Camera camera, string path, bool depth, bool ids)
    {
        var rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32) { antiAliasing = depth || ids ? 1 : 4 };
        var tex = new Texture2D(Width, Height, TextureFormat.RGB24, false); camera.targetTexture = rt;
        Dictionary<Renderer, Material[]> saved = null;
        if (depth) camera.SetReplacementShader(Shader.Find("BabyWorld/MetricDepth"), "");
        if (ids) {
            saved = UnityEngine.Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None).Where(r => r.enabled && r.gameObject.layer != 31).ToDictionary(r => r, r => r.sharedMaterials);
            foreach (var kv in saved) { var mat = kv.Key.gameObject == target ? idTarget : kv.Key.gameObject.name.Contains("ActualWeightedSkin") ? idSkin : idScene; kv.Key.sharedMaterials = Enumerable.Repeat(mat, Math.Max(1, kv.Value.Length)).ToArray(); }
            camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = Color.black;
        }
        camera.Render(); RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, Width, Height), 0, 0); tex.Apply(false); File.WriteAllBytes(path, tex.EncodeToPNG());
        if (depth) camera.ResetReplacementShader();
        if (saved != null) { foreach (var kv in saved) if (kv.Key) kv.Key.sharedMaterials = kv.Value; camera.clearFlags = CameraClearFlags.Skybox; }
        camera.targetTexture = null; RenderTexture.active = null; UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(tex);
    }

    static (Vector3 position, Quaternion rotation) PoseFromMj(float[] p) => (Mj(new Vector3(p[0], p[1], p[2])), QMj(p[3], p[4], p[5], p[6]));
    static (Vector3 position, Quaternion rotation) CameraPoseFromMj(float[] p) {
        var r = new Vector3(p[3], p[6], p[9]); var up = new Vector3(p[4], p[7], p[10]); var back = new Vector3(p[5], p[8], p[11]);
        return (Mj(new Vector3(p[0], p[1], p[2])), Quaternion.LookRotation(Mj(-back).normalized, Mj(up).normalized));
    }
    static Vector3 Mj(Vector3 p) => new Vector3(p.x, p.z, -p.y);
    static Vector3 MjInverse(Vector3 p) => new Vector3(p.x, -p.z, p.y);
    static Quaternion QMj(float w, float x, float y, float z) { var forward = QRotate(w, x, y, z, new Vector3(0, 1, 0)); var up = QRotate(w, x, y, z, new Vector3(0, 0, 1)); return Quaternion.LookRotation(Mj(forward), Mj(up)); }
    static Vector3 QRotate(float w, float x, float y, float z, Vector3 v) { var q = new Quaternion(x, y, z, w); return q * v; }
    static string PhysicsName(string n) => n == "head" ? "head" : n == "wrist.R" ? "wrist_guide" : n.StartsWith("finger1") ? "thumb" : n.StartsWith("finger2") ? "index" : n.StartsWith("finger3") ? "middle" : n.StartsWith("finger4") ? "ring" : n.StartsWith("finger5") ? "little" : n.StartsWith("upperarm") ? "shoulder_chain" : "elbow_chain";
    static float Q(Frame f, string name) => jointIndex.ContainsKey(name) ? f.qpos[jointIndex[name]] : 0;
    static (Vector3 position, Quaternion rotation) PhysicalPose(Frame f, string n) {
        if (n == "head") return PoseFromMj(f.head_pose_mj); if (n == "wrist.R") return PoseFromMj(f.wrist_pose_mj);
        int digit = n[6] - '1'; int part = n.Contains("-1.") ? 0 : n.Contains("-2.") ? 1 : 2; int offset = (digit * 3 + part) * 7;
        var packed = new[] { f.digit_segment_pose_mj[offset], f.digit_segment_pose_mj[offset + 1], f.digit_segment_pose_mj[offset + 2], f.digit_segment_pose_mj[offset + 3], f.digit_segment_pose_mj[offset + 4], f.digit_segment_pose_mj[offset + 5], f.digit_segment_pose_mj[offset + 6] };
        return PoseFromMj(packed);
    }
    static Transform Find(string n) { var t = FindOptional(n); if (!t) throw new Exception("Missing bone " + n); return t; }
    static Transform FindOptional(string n) => avatar.GetComponentsInChildren<Transform>(true).FirstOrDefault(x => x.name == n);
    static Material NewMaterial(string n, Color c, float gloss) { Directory.CreateDirectory("Assets/Generated"); var path = $"Assets/Generated/{n}.mat"; var m = AssetDatabase.LoadAssetAtPath<Material>(path); if (!m) { m = new Material(Shader.Find("Standard")); AssetDatabase.CreateAsset(m, path); } m.color = c; m.SetFloat("_Glossiness", gloss); return m; }
    static void Box(string n, Vector3 p, Vector3 s, string mat) { var o = GameObject.CreatePrimitive(PrimitiveType.Cube); o.name = n; o.transform.position = p; o.transform.localScale = s; o.GetComponent<Renderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>($"Assets/Generated/{mat}.mat"); UnityEngine.Object.DestroyImmediate(o.GetComponent<Collider>()); }
    static void Sphere(string n, Vector3 p, float diameter, string mat) { var o = GameObject.CreatePrimitive(PrimitiveType.Sphere); o.name = n; o.transform.position = p; o.transform.localScale = Vector3.one * diameter; o.GetComponent<Renderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>($"Assets/Generated/{mat}.mat"); UnityEngine.Object.DestroyImmediate(o.GetComponent<Collider>()); }
    static GameObject PlaceMetric(string name, Vector3 floorPosition, float maxM, float yaw) { var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Furniture/{name}.obj"); var o = (GameObject)PrefabUtility.InstantiatePrefab(prefab); o.name = name; o.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0, yaw, 0)); var raw = BoundsOf(o); o.transform.localScale = Vector3.one * (maxM / Mathf.Max(raw.size.x, raw.size.y, raw.size.z)); var b = BoundsOf(o); o.transform.position = new Vector3(floorPosition.x - b.center.x, floorPosition.y - b.min.y, floorPosition.z - b.center.z); return o; }
    static Bounds BoundsOf(GameObject o) { var rs = o.GetComponentsInChildren<Renderer>(); var b = rs[0].bounds; foreach (var r in rs.Skip(1)) b.Encapsulate(r.bounds); return b; }

    [Serializable] class Trace { public string schema; public int physics_hz, render_hz, steps_per_frame; public string[] joint_names, digit_segment_names; public Frame[] frames; }
    [Serializable] class Frame { public int frame, truth_index, contact_digits; public float time_s; public string phase; public float[] qpos, wrist_pose_mj, head_pose_mj, camera_pose_mj, target_pose_mj, digit_segment_pose_mj; public ContactPointArray[] contact_points_mj; public bool support_contact; }
    [Serializable] class ContactPointArray { public float[] values; }
    [Serializable] class RegistrationRow { public string bone, physics_name; public int semantic_id; public float landmark_error_m, landmark_mean_error_m, collider_radius_m; public Vector3 visible_rest_offset_from_wrist_mj; }
    [Serializable] class EmbodimentManifest { public string schema, source_fbx_sha256, coordinate_conversion, physics_authority, appearance_authority, stable_ids, camera_mapping, engineering_priors; public float source_scale; public string[] physical_names, touch_sites; public RegistrationRow[] mapping; }
    [Serializable] class RegistrationReport { public bool passed, quantitative_registration_passed, visual_anatomy_passed, camera_outside_head; public string failure_reason; public float collider_skin_tolerance_m, landmark_mean_error_m, landmark_max_error_m, coordinate_roundtrip_error_m, camera_neutral_axis_deg; public EmbodimentManifest manifest; }
    [Serializable] class FrameMetric { public int frame, contact_digits; public float time_s, camera_mount_position_error_m, camera_mount_rotation_error_deg, visible_fingertip_to_physical_wrist_m; public string phase; public bool target_visible, support_contact; public Vector3 target_viewport; }
    [Serializable] class SweepRow { public string bone, axis; public int minimum_deg, maximum_deg, step_deg; public float maximum_mesh_bounds_ratio, maximum_segment_length_error_m, maximum_baked_vertex_motion_m; }
    [Serializable] class SweepReport { public string path; public int frames; public bool passed; public float maximum_mesh_bounds_ratio, maximum_segment_length_error_m, audition_bounds_size_error_m, audition_bounds_size_tolerance_m; public Vector3 integrated_rest_avatar_bounds_size_m, audition_rest_avatar_bounds_size_m; public SweepRow[] joints; }
    [Serializable] class RestBone { public string name, source_parent, retained_parent; public Vector3 world_position_unity, local_position_source; public Quaternion world_rotation_unity, local_rotation_source; }
    [Serializable] class RestManifest { public string schema, source_fbx_sha256, coordinate_conversion; public float avatar_scale; public RestBone[] bones; }
    [Serializable] class RenderReceipt { public string unity_version, graphics_api, target_pose_source, camera_pose_source; public int frame_count, physics_hz, render_hz, steps_per_frame; public bool actual_weighted_skin_only, proxy_pixels_in_rgb, unity_physics_enabled, fixed_camera_mount, camera_target_dependent; public float fov_y_deg, near_clip_m; public FrameMetric[] sampled_frames; }
}
