using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class AuditionBuilder
{
    const int Width = 960, Height = 540, Fps = 30, Frames = 210;
    static readonly string Output = Environment.GetEnvironmentVariable("UNITY_AUDITION_OUTPUT");
    static readonly Vector3 RoomForward = Vector3.back;
    static readonly List<Transform> Ik = new List<Transform>();
    static readonly List<Transform> Fingers = new List<Transform>();
    static readonly Dictionary<Transform, Quaternion> Rest = new Dictionary<Transform, Quaternion>();
    static readonly List<ObjectMetric> ObjectMetrics = new List<ObjectMetric>();
    static readonly List<FrameMetric> FrameMetrics = new List<FrameMetric>();
    static readonly List<ProxySegment> ProxySegments = new List<ProxySegment>();
    static GameObject avatar;
    static Transform head, shoulder, wrist, fingertip;
    static Camera camera;
    static Vector3 touchPoint;
    static Bounds headWeightedBounds;
    static float armReach;
    static Transform proxyPalm;
    static SkinnedMeshRenderer skinRenderer;
    static Vector3[] restBakedVertices;
    static HashSet<int> rightArmWeightedVertices;
    static BoneAudit[] boneAudits;
    static DeformationAudit deformationAudit;
    static Mesh bakedVisibleMesh;

    [MenuItem("BabyWorld/Render Visual Audition")]
    public static void Render()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_AUDITION_OUTPUT is required");
        Directory.CreateDirectory(Output);
        BuildScene();
        var rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
        var tex = new Texture2D(Width, Height, TextureFormat.RGB24, false);
        camera.targetTexture = rt;
        var sampleFrames = new HashSet<int>(new[] { 0, 45, 90, 120, 150, 180, 209 });
        for (int frame = 0; frame < Frames; frame++) {
            Pose(frame / (float)Fps);
            UpdateFirstPersonProxy();
            UpdateBakedSkinVisual();
            if (sampleFrames.Contains(frame)) RecordFrame(frame);
            camera.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, Width, Height), 0, 0); tex.Apply(false);
            File.WriteAllBytes(Path.Combine(Output, $"frame_{frame:D4}.png"), tex.EncodeToPNG());
        }
        RenderTexture.active = null;
        var headClosest = headWeightedBounds.ClosestPoint(camera.transform.position);
        var diagnostics = new Diagnostics {
            editor = "6000.0.80f1", pipeline = "Built-in Render Pipeline", graphics_api = SystemInfo.graphicsDeviceType.ToString(),
            unit_convention = "1 Unity unit = 1 meter", kenney_normalization = "uniform scale = desired maximum world dimension / measured raw Renderer.bounds maximum dimension",
            camera_world_position = camera.transform.position, camera_world_forward = camera.transform.forward, camera_world_up = camera.transform.up,
            head_world_position = head.position, head_to_camera = camera.transform.position - head.position,
            room_forward = RoomForward, camera_room_forward_dot = Vector3.Dot(camera.transform.forward, RoomForward),
            camera_room_forward_angle_deg = Vector3.Angle(camera.transform.forward, RoomForward),
            mount_local_position = camera.transform.localPosition, mount_local_euler = camera.transform.localEulerAngles,
            vertical_fov_deg = camera.fieldOfView, near_clip_m = camera.nearClipPlane,
            head_weighted_bounds_center = headWeightedBounds.center, head_weighted_bounds_size = headWeightedBounds.size,
            camera_inside_head_weighted_bounds = headWeightedBounds.Contains(camera.transform.position),
            head_weighted_aabb_clearance_m = Vector3.Distance(headClosest, camera.transform.position),
            avatar_bounds = Metric("avatar", BoundsOf(avatar)), eye_height_m = camera.transform.position.y,
            shoulder_world_position = shoulder.position, wrist_rest_world_position = RestWorldPosition(wrist),
            articulated_arm_reach_m = armReach, target_reach_fraction = Vector3.Distance(shoulder.position, touchPoint) / armReach,
            target_world_position = touchPoint, objects = ObjectMetrics.ToArray(), frames = FrameMetrics.ToArray(),
            skin_renderer_enabled = skinRenderer.enabled, skin_update_when_offscreen = skinRenderer.updateWhenOffscreen,
            bound_bones = boneAudits, deformation = deformationAudit,
            disclosure = "All motion, CCD IK, finger closure, and touch are kinematic/nonphysical; target is static. The POV arm/hand is a smooth bone-driven first-person proxy because the imported one-piece skinned mesh did not visibly deform under batch bone writes. Visual feasibility only."
        };
        File.WriteAllText(Path.Combine(Output, "diagnostics.json"), JsonUtility.ToJson(diagnostics, true));
        AssetDatabase.SaveAssets(); EditorApplication.Exit(0);
    }

    static void BuildScene()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        Ik.Clear(); Fingers.Clear(); Rest.Clear(); ObjectMetrics.Clear(); FrameMetrics.Clear(); ProxySegments.Clear();
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(.52f, .61f, .72f);
        RenderSettings.ambientEquatorColor = new Color(.34f, .30f, .27f);
        RenderSettings.ambientGroundColor = new Color(.14f, .12f, .10f);
        RenderSettings.fog = false;

        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx"));
        avatar.name = "CC0_Child_Avatar";
        avatar.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        avatar.transform.localScale = Vector3.one * 1.9f; // 0.656 m CC0 source -> 1.246 m child shell.
        foreach (var r in avatar.GetComponentsInChildren<SkinnedMeshRenderer>()) {
            r.updateWhenOffscreen = true;
            r.localBounds = new Bounds(Vector3.zero, Vector3.one * 4f);
            var skin = new Material(Shader.Find("Standard")); skin.color = new Color(.74f, .49f, .36f); skin.SetFloat("_Glossiness", .18f); r.sharedMaterial = skin;
        }
        head = Find("head"); shoulder = Find("upperarm01.R"); wrist = Find("wrist.R"); fingertip = Find("finger2-3.R");
        foreach (var n in new[] { "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R" }) Ik.Add(Find(n));
        foreach (var t in avatar.GetComponentsInChildren<Transform>()) if (t.name.StartsWith("finger") && t.name.EndsWith(".R")) Fingers.Add(t);
        foreach (var t in avatar.GetComponentsInChildren<Transform>()) Rest[t] = t.localRotation;
        armReach = ChainLength(new[] { "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R", "finger2-1.R", "finger2-2.R", "finger2-3.R" });
        AuditSkinBindingAtRest();
        headWeightedBounds = HeadWeightedBounds();
        BuildFirstPersonProxy();
        BuildBakedSkinVisual();

        // Build once in world meters from the rest head geometry and fixed root/room basis, then freeze under head.
        var cameraWorld = new Vector3(headWeightedBounds.center.x, headWeightedBounds.center.y + .018f, headWeightedBounds.min.z - .032f);
        var camGo = new GameObject("FrozenHeadCameraMount");
        var fixedViewDirection = new Vector3(0, -Mathf.Sin(50 * Mathf.Deg2Rad), -Mathf.Cos(50 * Mathf.Deg2Rad));
        camGo.transform.SetPositionAndRotation(cameraWorld, Quaternion.LookRotation(fixedViewDirection, Vector3.up));
        camGo.transform.SetParent(head, true);
        camera = camGo.AddComponent<Camera>(); camera.fieldOfView = 62; camera.nearClipPlane = .03f; camera.farClipPlane = 20; camera.clearFlags = CameraClearFlags.Skybox;
        camera.cullingMask &= ~(1 << 31); // Exclude only the duplicate source draw; the baked mesh is the same full skinned body.

        Mat("Floor", new Color(.43f, .29f, .18f)); Box("Floor", new Vector3(0, -.035f, -1.8f), new Vector3(4.8f, .07f, 5.4f), "Floor");
        Mat("Wall", new Color(.74f, .79f, .75f)); Box("BackWall", new Vector3(0, 1.35f, -4.45f), new Vector3(4.8f, 2.7f, .08f), "Wall"); Box("SideWall", new Vector3(-2.36f, 1.35f, -1.8f), new Vector3(.08f, 2.7f, 5.4f), "Wall");
        Mat("Trim", new Color(.94f, .90f, .80f)); Box("Baseboard", new Vector3(0, .09f, -4.36f), new Vector3(4.7f, .18f, .08f), "Trim");
        PlaceMetric("rugRectangle", new Vector3(.1f, 0, -1.9f), 2.35f, 0);
        PlaceMetric("loungeSofaLong", new Vector3(1.05f, 0, -3.72f), 2.05f, 0);
        PlaceMetric("bookcaseOpen", new Vector3(-1.82f, 0, -3.68f), 1.75f, 0);
        PlaceMetric("pottedPlant", new Vector3(-1.68f, 0, -2.62f), 1.05f, 180);
        PlaceMetric("lampSquareFloor", new Vector3(1.9f, 0, -3.42f), 1.65f, 180);
        var table = PlaceMetric("tableCoffee", new Vector3(.2f, 0, -.3f), 1.35f, 180);
        PlaceMetric("chairCushion", new Vector3(-1.12f, 0, -1.55f), .82f, 205);
        var tableBounds = BoundsOf(table);
        table.transform.localScale = new Vector3(table.transform.localScale.x, table.transform.localScale.y * (.62f / tableBounds.size.y), table.transform.localScale.z);
        tableBounds = BoundsOf(table); table.transform.position += Vector3.up * -tableBounds.min.y; tableBounds = BoundsOf(table);
        float desiredDistance = armReach * .85f;
        float targetX = shoulder.position.x - .10f;
        float targetY = tableBounds.max.y + .075f;
        float planarZ = Mathf.Sqrt(Mathf.Max(.01f, desiredDistance * desiredDistance - Mathf.Pow(targetX - shoulder.position.x, 2) - Mathf.Pow(targetY - shoulder.position.y, 2)));
        touchPoint = new Vector3(targetX, targetY, shoulder.position.z - planarZ);
        table.transform.position += new Vector3(touchPoint.x + .18f - tableBounds.center.x, 0, touchPoint.z - .12f - tableBounds.center.z);
        ObjectMetrics.RemoveAt(ObjectMetrics.Count - 2); // Replace the pre-translation table metric; chair is the last record.
        ObjectMetrics.Insert(ObjectMetrics.Count - 1, Metric("tableCoffee", BoundsOf(table), tableBounds.size / table.transform.localScale.x, table.transform.localScale.x));
        tableBounds = BoundsOf(table);
        PlaceMetric("books", new Vector3(tableBounds.center.x + .22f, tableBounds.max.y, tableBounds.center.z - .08f), .28f, 192);
        Mat("Toy", new Color(.92f, .26f, .12f)); var toy = GameObject.CreatePrimitive(PrimitiveType.Sphere); toy.name = "NearbyRedToy"; toy.transform.position = touchPoint; toy.transform.localScale = Vector3.one * .15f; toy.GetComponent<Renderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>("Assets/Generated/Toy.mat");
        ObjectMetrics.Add(Metric(toy.name, BoundsOf(toy)));

        var sun = new GameObject("WindowKey").AddComponent<Light>(); sun.type = LightType.Directional; sun.intensity = 1.25f; sun.color = new Color(1f, .90f, .76f); sun.transform.rotation = Quaternion.Euler(42, -28, 0); sun.shadows = LightShadows.Soft;
        var fill = new GameObject("WarmFill").AddComponent<Light>(); fill.type = LightType.Point; fill.range = 6; fill.intensity = 3.5f; fill.color = new Color(1f, .74f, .52f); fill.transform.position = new Vector3(1.4f, 2.1f, -1.1f); fill.shadows = LightShadows.Soft;
    }

    static void Pose(float t)
    {
        foreach (var kv in Rest) if (kv.Key) kv.Key.localRotation = kv.Value;
        float look = Smooth(0, 1.35f, t) - Smooth(1.35f, 2.15f, t);
        head.localRotation = Rest[head] * Quaternion.Euler(7f * look, 13f * look, -1.5f * look);
        float amount = Smooth(1.35f, 3.65f, t) * (1 - Smooth(4.65f, 6.55f, t));
        Vector3 target = Vector3.Lerp(fingertip.position, touchPoint, amount);
        for (int pass = 0; pass < 18; pass++) for (int i = Ik.Count - 1; i >= 0; i--) {
            var b = Ik[i]; var delta = Quaternion.FromToRotation(fingertip.position - b.position, target - b.position);
            b.rotation = Quaternion.Slerp(Quaternion.identity, delta, .56f) * b.rotation;
        }
        float close = Smooth(3.05f, 3.85f, t) * (1 - Smooth(4.35f, 5.05f, t));
        foreach (var f in Fingers) { int segment = f.name.Contains("-1.") ? 16 : f.name.Contains("-2.") ? 26 : 32; f.localRotation = Rest[f] * Quaternion.Euler(segment * close, 0, 0); }
    }

    static void RecordFrame(int frame)
    {
        if (frame == 120) AuditContactDeformation();
        FrameMetrics.Add(new FrameMetric {
            frame = frame, time_s = frame / (float)Fps, camera_position = camera.transform.position,
            camera_forward = camera.transform.forward, target_viewport = camera.WorldToViewportPoint(touchPoint),
            target_visible = Visible(camera.WorldToViewportPoint(touchPoint)), wrist_position = wrist.position, fingertip_viewport = camera.WorldToViewportPoint(fingertip.position),
            wrist_to_touch_error_m = Vector3.Distance(wrist.position, touchPoint), fingertip_to_touch_error_m = Vector3.Distance(fingertip.position, touchPoint), head_weighted_geometry_contains_camera = HeadWeightedBounds().Contains(camera.transform.position)
        });
    }

    static void AuditSkinBindingAtRest()
    {
        skinRenderer = avatar.GetComponentsInChildren<SkinnedMeshRenderer>().Single(); skinRenderer.enabled = true; skinRenderer.updateWhenOffscreen = true;
        var auditedNames = new[] { "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R", "finger1-1.R", "finger2-1.R", "finger2-2.R", "finger2-3.R", "finger3-1.R", "finger4-1.R", "finger5-1.R" };
        var sourceWeights = skinRenderer.sharedMesh.boneWeights; rightArmWeightedVertices = new HashSet<int>(); var audits = new List<BoneAudit>();
        foreach (var name in auditedNames) {
            var driven = Find(name); int boneIndex = Array.IndexOf(skinRenderer.bones, driven); int vertices = 0; float totalWeight = 0;
            if (boneIndex >= 0) for (int i = 0; i < sourceWeights.Length; i++) { var w = sourceWeights[i]; float weight = 0; if (w.boneIndex0 == boneIndex) weight += w.weight0; if (w.boneIndex1 == boneIndex) weight += w.weight1; if (w.boneIndex2 == boneIndex) weight += w.weight2; if (w.boneIndex3 == boneIndex) weight += w.weight3; if (weight > 0) { vertices++; totalWeight += weight; rightArmWeightedVertices.Add(i); } }
            audits.Add(new BoneAudit { name = name, renderer_bone_index = boneIndex, exact_transform_instance = boneIndex >= 0 && ReferenceEquals(skinRenderer.bones[boneIndex], driven), weighted_vertex_count = vertices, total_vertex_weight = totalWeight });
        }
        boneAudits = audits.ToArray(); var baked = new Mesh(); skinRenderer.BakeMesh(baked, true); restBakedVertices = baked.vertices; UnityEngine.Object.DestroyImmediate(baked);
    }

    static void AuditContactDeformation()
    {
        var baked = new Mesh(); skinRenderer.BakeMesh(baked, true); var contact = baked.vertices; float allSum = 0, allMax = 0, armSum = 0, armMax = 0; int armCount = 0; bool armInit = false; Bounds armBounds = new Bounds();
        for (int i = 0; i < contact.Length && i < restBakedVertices.Length; i++) { Vector3 contactWorld = skinRenderer.transform.TransformPoint(contact[i]); float d = Vector3.Distance(contactWorld, skinRenderer.transform.TransformPoint(restBakedVertices[i])); allSum += d; allMax = Mathf.Max(allMax, d); if (rightArmWeightedVertices.Contains(i)) { armSum += d; armMax = Mathf.Max(armMax, d); armCount++; if (!armInit) { armBounds = new Bounds(contactWorld, Vector3.zero); armInit = true; } else armBounds.Encapsulate(contactWorld); } }
        deformationAudit = new DeformationAudit { vertex_count = contact.Length, right_arm_weighted_vertex_count = armCount, mean_all_vertex_displacement_m = allSum / Mathf.Max(1, contact.Length), max_all_vertex_displacement_m = allMax, mean_right_arm_vertex_displacement_m = armSum / Mathf.Max(1, armCount), max_right_arm_vertex_displacement_m = armMax, contact_right_arm_bounds_center = armBounds.center, contact_right_arm_bounds_size = armBounds.size, contact_right_arm_bounds_distance_to_target_m = Vector3.Distance(armBounds.ClosestPoint(touchPoint), touchPoint) };
        UnityEngine.Object.DestroyImmediate(baked);
    }

    static void BuildFirstPersonProxy()
    {
        var proxyMat = new Material(Shader.Find("Standard")); proxyMat.color = new Color(.78f, .52f, .38f); proxyMat.SetFloat("_Glossiness", .28f);
        AddProxy("upper_arm_a", "upperarm01.R", "upperarm02.R", .042f, false, proxyMat);
        AddProxy("upper_arm_b", "upperarm02.R", "lowerarm01.R", .040f, false, proxyMat);
        AddProxy("forearm_a", "lowerarm01.R", "lowerarm02.R", .034f, false, proxyMat);
        AddProxy("forearm_b", "lowerarm02.R", "wrist.R", .029f, false, proxyMat);
        foreach (var finger in new[] { "1", "2", "3", "4", "5" }) {
            float radius = finger == "1" ? .0115f : .009f;
            AddProxy($"finger{finger}_proximal", $"finger{finger}-1.R", $"finger{finger}-2.R", radius, false, proxyMat);
            AddProxy($"finger{finger}_middle", $"finger{finger}-2.R", $"finger{finger}-3.R", radius * .9f, false, proxyMat);
            AddProxy($"finger{finger}_distal", $"finger{finger}-2.R", $"finger{finger}-3.R", radius * .78f, true, proxyMat);
        }
        var palm = GameObject.CreatePrimitive(PrimitiveType.Sphere); palm.name = "POV_Palm"; palm.GetComponent<Renderer>().sharedMaterial = proxyMat; proxyPalm = palm.transform;
        UpdateFirstPersonProxy();
    }

    static void BuildBakedSkinVisual()
    {
        var o = new GameObject("ActualBakedSkinnedChildPOV"); o.transform.SetParent(skinRenderer.transform.parent, false);
        o.transform.localPosition = skinRenderer.transform.localPosition; o.transform.localRotation = skinRenderer.transform.localRotation; o.transform.localScale = skinRenderer.transform.localScale;
        bakedVisibleMesh = new Mesh { name = "ActualBakedSkinnedChildFrame" }; o.AddComponent<MeshFilter>().sharedMesh = bakedVisibleMesh;
        var renderer = o.AddComponent<MeshRenderer>(); renderer.sharedMaterials = skinRenderer.sharedMaterials; renderer.enabled = true;
        skinRenderer.gameObject.layer = 31;
        foreach (var s in ProxySegments) s.visual.GetComponent<Renderer>().enabled = false;
        proxyPalm.GetComponent<Renderer>().enabled = false;
        UpdateBakedSkinVisual();
    }

    static void UpdateBakedSkinVisual()
    {
        bakedVisibleMesh.Clear(); skinRenderer.BakeMesh(bakedVisibleMesh, true); bakedVisibleMesh.RecalculateBounds();
    }

    static void AddProxy(string name, string a, string b, float radius, bool terminal, Material material)
    {
        var o = GameObject.CreatePrimitive(PrimitiveType.Capsule); o.name = "POV_" + name; o.GetComponent<Renderer>().sharedMaterial = material;
        ProxySegments.Add(new ProxySegment { visual = o.transform, a = Find(a), b = Find(b), radius = radius, terminal = terminal });
    }

    static void UpdateFirstPersonProxy()
    {
        foreach (var s in ProxySegments) {
            Vector3 a = s.terminal ? s.b.position : s.a.position;
            Vector3 b = s.terminal ? s.b.position + (s.b.position - s.a.position) * .72f : s.b.position;
            Vector3 delta = b - a; float length = Mathf.Max(delta.magnitude, s.radius * 2.2f);
            s.visual.position = (a + b) * .5f; s.visual.rotation = Quaternion.FromToRotation(Vector3.up, delta.normalized);
            s.visual.localScale = new Vector3(s.radius * 2, length * .5f, s.radius * 2);
        }
        var knuckles = new[] { Find("finger2-1.R"), Find("finger3-1.R"), Find("finger4-1.R") };
        Vector3 palmEnd = knuckles.Select(x => x.position).Aggregate(Vector3.zero, (a, b) => a + b) / knuckles.Length;
        Vector3 palmDirection = palmEnd - wrist.position; proxyPalm.position = (wrist.position + palmEnd) * .5f;
        proxyPalm.rotation = Quaternion.LookRotation(palmDirection.normalized, Vector3.up); proxyPalm.localScale = new Vector3(.082f, .038f, Mathf.Max(.075f, palmDirection.magnitude));
    }

    static bool Visible(Vector3 p) => p.z > camera.nearClipPlane && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
    static float Smooth(float a, float b, float t) => Mathf.SmoothStep(0, 1, Mathf.InverseLerp(a, b, t));
    static Transform Find(string n) { var t = avatar.GetComponentsInChildren<Transform>(true).FirstOrDefault(x => x.name == n); if (!t) throw new Exception("Missing bone " + n); return t; }
    static Vector3 RestWorldPosition(Transform t) { return t.position; }
    static float ChainLength(string[] names) { float sum = 0; for (int i = 1; i < names.Length; i++) sum += Vector3.Distance(Find(names[i - 1]).position, Find(names[i]).position); return sum + Vector3.Distance(Find(names.Last()).position, fingertip.position); }

    static GameObject PlaceMetric(string name, Vector3 floorPosition, float desiredMaxDimensionM, float yaw)
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Furniture/{name}.obj"); if (!prefab) throw new Exception("Missing furniture " + name);
        var o = (GameObject)PrefabUtility.InstantiatePrefab(prefab); o.name = name; o.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0, yaw, 0)); o.transform.localScale = Vector3.one;
        var raw = BoundsOf(o); float scale = desiredMaxDimensionM / Mathf.Max(raw.size.x, raw.size.y, raw.size.z); o.transform.localScale = Vector3.one * scale;
        var scaled = BoundsOf(o); o.transform.position = new Vector3(floorPosition.x - scaled.center.x, floorPosition.y - scaled.min.y, floorPosition.z - scaled.center.z);
        ObjectMetrics.Add(Metric(name, BoundsOf(o), raw.size, scale)); return o;
    }

    static Bounds BoundsOf(GameObject o)
    {
        var rs = o.GetComponentsInChildren<Renderer>(); if (rs.Length == 0) throw new Exception("No renderer on " + o.name);
        var b = rs[0].bounds; for (int i = 1; i < rs.Length; i++) b.Encapsulate(rs[i].bounds); return b;
    }

    static Bounds HeadWeightedBounds()
    {
        bool initialized = false; Bounds result = new Bounds();
        foreach (var smr in avatar.GetComponentsInChildren<SkinnedMeshRenderer>()) {
            var source = smr.sharedMesh; var weights = source.boneWeights; var baked = new Mesh(); smr.BakeMesh(baked, true); var vertices = baked.vertices;
            var headBoneIndices = new HashSet<int>(); for (int i = 0; i < smr.bones.Length; i++) if (smr.bones[i] == head || smr.bones[i].IsChildOf(head)) headBoneIndices.Add(i);
            for (int i = 0; i < vertices.Length && i < weights.Length; i++) {
                var w = weights[i]; float headWeight = 0;
                if (headBoneIndices.Contains(w.boneIndex0)) headWeight += w.weight0; if (headBoneIndices.Contains(w.boneIndex1)) headWeight += w.weight1;
                if (headBoneIndices.Contains(w.boneIndex2)) headWeight += w.weight2; if (headBoneIndices.Contains(w.boneIndex3)) headWeight += w.weight3;
                if (headWeight < .5f) continue; var world = smr.transform.TransformPoint(vertices[i]);
                if (!initialized) { result = new Bounds(world, Vector3.zero); initialized = true; } else result.Encapsulate(world);
            }
            UnityEngine.Object.DestroyImmediate(baked);
        }
        if (!initialized) throw new Exception("No head-weighted vertices found"); return result;
    }

    static ObjectMetric Metric(string name, Bounds b, Vector3 raw = default, float scale = 1) => new ObjectMetric { name = name, center = b.center, size = b.size, min = b.min, max = b.max, raw_size = raw, applied_uniform_scale = scale };
    static void Mat(string name, Color color) { Directory.CreateDirectory("Assets/Generated"); var path = $"Assets/Generated/{name}.mat"; var m = AssetDatabase.LoadAssetAtPath<Material>(path); if (!m) { m = new Material(Shader.Find("Standard")); AssetDatabase.CreateAsset(m, path); } m.color = color; m.SetFloat("_Glossiness", .25f); }
    static void Box(string n, Vector3 p, Vector3 s, string mat) { var o = GameObject.CreatePrimitive(PrimitiveType.Cube); o.name = n; o.transform.position = p; o.transform.localScale = s; o.GetComponent<Renderer>().sharedMaterial = AssetDatabase.LoadAssetAtPath<Material>($"Assets/Generated/{mat}.mat"); ObjectMetrics.Add(Metric(n, BoundsOf(o))); }

    [Serializable] class Diagnostics { public string editor, pipeline, graphics_api, unit_convention, kenney_normalization, disclosure; public Vector3 camera_world_position, camera_world_forward, camera_world_up, head_world_position, head_to_camera, room_forward, mount_local_position, mount_local_euler, head_weighted_bounds_center, head_weighted_bounds_size, shoulder_world_position, wrist_rest_world_position, target_world_position; public float camera_room_forward_dot, camera_room_forward_angle_deg, vertical_fov_deg, near_clip_m, head_weighted_aabb_clearance_m, eye_height_m, articulated_arm_reach_m, target_reach_fraction; public bool camera_inside_head_weighted_bounds, skin_renderer_enabled, skin_update_when_offscreen; public ObjectMetric avatar_bounds; public ObjectMetric[] objects; public FrameMetric[] frames; public BoneAudit[] bound_bones; public DeformationAudit deformation; }
    [Serializable] class ObjectMetric { public string name; public Vector3 center, size, min, max, raw_size; public float applied_uniform_scale; }
    [Serializable] class FrameMetric { public int frame; public float time_s, wrist_to_touch_error_m, fingertip_to_touch_error_m; public Vector3 camera_position, camera_forward, target_viewport, fingertip_viewport, wrist_position; public bool target_visible, head_weighted_geometry_contains_camera; }
    [Serializable] class BoneAudit { public string name; public int renderer_bone_index, weighted_vertex_count; public bool exact_transform_instance; public float total_vertex_weight; }
    [Serializable] class DeformationAudit { public int vertex_count, right_arm_weighted_vertex_count; public float mean_all_vertex_displacement_m, max_all_vertex_displacement_m, mean_right_arm_vertex_displacement_m, max_right_arm_vertex_displacement_m, contact_right_arm_bounds_distance_to_target_m; public Vector3 contact_right_arm_bounds_center, contact_right_arm_bounds_size; }
    class ProxySegment { public Transform visual, a, b; public float radius; public bool terminal; }
}
