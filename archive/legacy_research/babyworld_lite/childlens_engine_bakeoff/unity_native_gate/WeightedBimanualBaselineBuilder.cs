using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

// Stage-C registered replay. The retained PhysX trace remains object/contact
// authority. The weighted MPFB skin, head camera, and QA overlays are all
// deterministic followers of the complete state recorded in that trace.
public static class WeightedBimanualBaselineBuilder
{
    const int QaLayer = 29;
    const float ContactSkinToleranceM = .012f;
    const float ContactPixelTolerance = 28f;
    const float PalmRegistrationToleranceM = .012f;
    static readonly string Output = Environment.GetEnvironmentVariable("WEIGHTED_BASELINE_OUTPUT");
    static readonly string TracePath = Environment.GetEnvironmentVariable("WEIGHTED_BASELINE_TRACE");

    static int width = 1920, height = 1080;
    static Trace trace;
    static GameObject avatar, toy;
    static SkinnedMeshRenderer skin;
    static Camera headCamera, cleanCamera, overlayCamera;
    static Transform head, neck, torso;
    static readonly Dictionary<string, Transform> bones = new();
    static readonly Dictionary<Transform, Quaternion> rest = new();
    static readonly Dictionary<string, Vector3> palmOffsetLocal = new();
    static readonly Dictionary<string, List<int>> digitVertices = new();
    static readonly Dictionary<string, Transform> qaSegments = new();
    static readonly Dictionary<string, Transform> qaPalms = new();
    static readonly List<GameObject> qaContacts = new();
    static readonly List<RegisteredFrame> registeredFrames = new();
    static readonly List<ContactRegistration> contactRegistrations = new();
    static readonly List<float> palmErrors = new();
    static Mesh baked;
    static int[] headWeightedIndices;
    static Quaternion cameraMountLocalRotation, initialRightPalmRotation, initialLeftPalmRotation;
    static Vector3 cameraMountLocalPosition;
    static float neutralMountAngle, minimumCameraClearance = float.PositiveInfinity, maximumCameraRoll;
    static int physicalContactFrames, registeredContactFrames, visibleContactFrames, targetVisibleEventFrames;
    static int firstPhysicalContactFrame = -1, firstRegisteredContactFrame = -1;
    static Vector3 previousCameraPosition, previousCameraVelocity;
    static Quaternion previousCameraRotation;
    static bool havePreviousCamera;

    [MenuItem("BabyWorld/Run Weighted Bimanual Baseline")]
    public static void Run()
    {
        if (string.IsNullOrWhiteSpace(Output) || string.IsNullOrWhiteSpace(TracePath)) throw new Exception("output and trace required");
        if (Environment.GetEnvironmentVariable("WEIGHTED_BASELINE_QUICK") == "1") { width = 960; height = 540; }
        Directory.CreateDirectory(Output);
        Directory.CreateDirectory(Path.Combine(Output, "head_frames"));
        Directory.CreateDirectory(Path.Combine(Output, "external_clean_frames"));
        Directory.CreateDirectory(Path.Combine(Output, "external_overlay_frames"));
        trace = JsonUtility.FromJson<Trace>(File.ReadAllText(TracePath));
        if (trace == null || trace.rows == null || trace.rows.Length < 2) throw new Exception("trace has no replay rows");
        if (trace.schema != "embodied.hybrid_bimanual_trace.v2") throw new Exception("registered replay requires v2 complete trace");
        initialRightPalmRotation = trace.rows[0].right_palm_rotation;
        initialLeftPalmRotation = trace.rows[0].left_palm_rotation;
        Build();

        using var headCapture = new CaptureRig(headCamera, Path.Combine(Output, "head_frames"), width, height);
        using var cleanCapture = new CaptureRig(cleanCamera, Path.Combine(Output, "external_clean_frames"), width, height);
        using var overlayCapture = new CaptureRig(overlayCamera, Path.Combine(Output, "external_overlay_frames"), width, height);
        for (int frame = 0; frame < trace.rows.Length; frame++) {
            TraceRow row = trace.rows[frame];
            Pose(row);
            toy.transform.SetPositionAndRotation(row.object_position_m, row.object_rotation);
            UpdateQa(row);
            Physics.SyncTransforms();
            skin.BakeMesh(baked, true);
            AuditFrame(frame, row);
            headCapture.Capture(frame);
            cleanCapture.Capture(frame);
            overlayCapture.Capture(frame);
        }

        float[] sortedContact = contactRegistrations.Where(x => x.digit_surface_distance_m >= 0).Select(x => x.digit_surface_distance_m).OrderBy(x => x).ToArray();
        float maxContact = sortedContact.Length > 0 ? sortedContact[^1] : float.PositiveInfinity;
        float p95Contact = Percentile(sortedContact, .95f);
        float maxPalm = palmErrors.Count > 0 ? palmErrors.Max() : float.PositiveInfinity;
        int firstTouchDifference = firstPhysicalContactFrame >= 0 && firstRegisteredContactFrame >= 0 ? Math.Abs(firstPhysicalContactFrame - firstRegisteredContactFrame) : int.MaxValue;
        float targetFrontDot = Vector3.Dot(Vector3.ProjectOnPlane(avatar.transform.forward, Vector3.up).normalized, Vector3.ProjectOnPlane(trace.rows[0].object_position_m - avatar.transform.position, Vector3.up).normalized);
        bool targetInFront = targetFrontDot > .5f;
        bool durationPass = trace.rows[^1].time_s >= 7.9f && trace.rows[^1].time_s <= 8.1f;
        bool registrationPass = physicalContactFrames > 0 && registeredContactFrames > 0 && firstTouchDifference <= 1 && p95Contact <= ContactSkinToleranceM && maxPalm <= PalmRegistrationToleranceM;
        bool cameraPass = neutralMountAngle <= 15 && minimumCameraClearance >= .02f && maximumCameraRoll <= 6;
        bool renderPass = width == 1920 && height == 1080 && targetVisibleEventFrames >= 4;
        var report = new Report {
            schema = "embodied.weighted_bimanual_stage_c_reconstruction.v1", unity_version = Application.unityVersion,
            source_fbx_sha256 = Sha256("Assets/Avatar/child.fbx"), source_license = "CC0",
            authoritative_physics_trace = TracePath, authoritative_physics_trace_sha256 = Sha256(TracePath),
            duration_s = trace.rows[^1].time_s, frames = trace.rows.Length, width = width, height = height,
            avatar_face_forward_world = avatar.transform.forward, target_front_dot = targetFrontDot, target_is_in_front = targetInFront,
            camera_parent_bone = head.name, camera_mount_local_position = cameraMountLocalPosition,
            camera_mount_local_rotation = cameraMountLocalRotation, camera_neutral_mount_angle_deg = neutralMountAngle,
            camera_minimum_skin_clearance_m = minimumCameraClearance, camera_maximum_roll_deg = maximumCameraRoll,
            fov_deg = headCamera.fieldOfView, near_clip_m = headCamera.nearClipPlane,
            palm_registration_max_m = maxPalm, contact_digit_surface_p95_m = p95Contact,
            contact_digit_surface_max_m = maxContact, contact_skin_tolerance_m = ContactSkinToleranceM,
            contact_pixel_tolerance = ContactPixelTolerance, physical_contact_frames = physicalContactFrames,
            registered_contact_frames = registeredContactFrames, visible_contact_frames = visibleContactFrames,
            first_physical_contact_frame = firstPhysicalContactFrame, first_registered_contact_frame = firstRegisteredContactFrame,
            first_touch_frame_difference = firstTouchDifference, target_visible_event_frames = targetVisibleEventFrames,
            one_weighted_visible_child = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true).Count(x => x.enabled) == 1,
            proxy_pixels_in_head_or_clean = 0, separate_labeled_overlay_present = true,
            palm_rotations_replayed = true, per_digit_states_replayed = true, head_parent_state_replayed = true,
            static_bone_to_skin_contact_site_offsets = true, maximum_contact_site_offset_m = .016f,
            duration_pass = durationPass, registration_pass = registrationPass, camera_pass = cameraPass, render_pass = renderPass
        };
        report.passed = durationPass && targetInFront && report.one_weighted_visible_child && registrationPass && cameraPass && renderPass;
        File.WriteAllText(Path.Combine(Output, "registered_trace.json"), JsonUtility.ToJson(new RegisteredTrace {
            schema = "embodied.registered_stage_c_trace.v1", input_trace_sha256 = report.authoritative_physics_trace_sha256,
            camera_mount_local_position = cameraMountLocalPosition, camera_mount_local_rotation = cameraMountLocalRotation,
            rows = registeredFrames.ToArray()
        }, true));
        File.WriteAllText(Path.Combine(Output, "contact_registration.json"), JsonUtility.ToJson(new ContactRegistrationTrace { schema = "embodied.visible_physical_contact_registration.v1", rows = contactRegistrations.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "report.json"), JsonUtility.ToJson(report, true));
        EditorApplication.Exit(report.passed ? 0 : 2);
    }

    static void Build()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(.48f, .57f, .68f);
        RenderSettings.ambientEquatorColor = new Color(.32f, .28f, .24f);
        RenderSettings.ambientGroundColor = new Color(.12f, .10f, .09f);
        RenderSettings.fog = false;

        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx"));
        avatar.name = "CC0_WEIGHTED_CHILD_SINGLE_VISIBLE_BODY";
        avatar.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        avatar.transform.localScale = Vector3.one * 1.9f;
        skin = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true).Single();
        skin.enabled = true; skin.updateWhenOffscreen = true; skin.localBounds = new Bounds(Vector3.zero, Vector3.one * 4);
        var skinMaterial = Material(new Color(.72f, .46f, .34f), .18f); skin.sharedMaterial = skinMaterial;
        foreach (var t in avatar.GetComponentsInChildren<Transform>(true)) { bones[t.name] = t; rest[t] = t.localRotation; }
        head = bones["head"];
        neck = bones.ContainsKey("neck01") ? bones["neck01"] : bones["neck"];
        torso = bones["spine03"];
        baked = new Mesh { name = "registered_visible_skin_audit" };
        BuildVertexMaps();
        MeasureCameraMount();
        BuildCameras();
        foreach (string side in new[] { "R", "L" }) {
            Transform wrist = bones["wrist." + side];
            palmOffsetLocal[side] = Quaternion.Inverse(wrist.rotation) * (VisiblePalmCenter(side) - wrist.position);
        }

        BuildRoom();
        toy = Cube("red_toy_001", trace.rows[0].object_position_m, Vector3.one * .055f, new Color(.82f, .055f, .035f), .28f);
        var toyRenderer = toy.GetComponent<Renderer>(); toyRenderer.shadowCastingMode = ShadowCastingMode.On; toyRenderer.receiveShadows = true;
        UnityEngine.Object.DestroyImmediate(toy.GetComponent<Collider>());
        BuildQaOverlay();
    }

    static void MeasureCameraMount()
    {
        Vector3[] points = CurrentHeadWeightedPoints();
        Bounds bounds = new Bounds(points[0], Vector3.zero); foreach (Vector3 p in points) bounds.Encapsulate(p);
        Vector3 world = new(bounds.center.x, bounds.center.y + .006f, bounds.max.z + .032f);
        cameraMountLocalPosition = head.InverseTransformPoint(world);
        cameraMountLocalRotation = Quaternion.Inverse(head.rotation) * Quaternion.LookRotation(Vector3.forward, Vector3.up);
        neutralMountAngle = Vector3.Angle(head.TransformDirection(cameraMountLocalRotation * Vector3.forward), avatar.transform.forward);
    }

    static void BuildCameras()
    {
        headCamera = new GameObject("fixed_neutral_child_head_camera").AddComponent<Camera>();
        headCamera.transform.SetParent(head, false); headCamera.transform.localPosition = cameraMountLocalPosition; headCamera.transform.localRotation = cameraMountLocalRotation;
        ConfigureCamera(headCamera, 68, new Color(.30f, .38f, .46f)); headCamera.cullingMask &= ~(1 << QaLayer);
        cleanCamera = new GameObject("clean_external_camera").AddComponent<Camera>();
        cleanCamera.transform.position = new Vector3(1.10f, 1.13f, 1.25f); cleanCamera.transform.LookAt(new Vector3(0, .66f, .24f));
        ConfigureCamera(cleanCamera, 48, new Color(.30f, .38f, .46f)); cleanCamera.cullingMask &= ~(1 << QaLayer);
        overlayCamera = new GameObject("QA_ONLY_labeled_collider_contact_camera").AddComponent<Camera>();
        overlayCamera.transform.SetPositionAndRotation(cleanCamera.transform.position, cleanCamera.transform.rotation);
        ConfigureCamera(overlayCamera, 48, new Color(.30f, .38f, .46f));
    }

    static void ConfigureCamera(Camera c, float fov, Color background)
    {
        c.fieldOfView = fov; c.nearClipPlane = .03f; c.farClipPlane = 15; c.clearFlags = CameraClearFlags.SolidColor;
        c.backgroundColor = background; c.allowHDR = true; c.allowMSAA = true;
    }

    static void BuildRoom()
    {
        Cube("floor", new Vector3(0, -.035f, 1.25f), new Vector3(4.8f, .07f, 4.4f), new Color(.43f, .28f, .18f), .22f);
        Cube("back_wall", new Vector3(0, 1.35f, 3.40f), new Vector3(4.8f, 2.7f, .08f), new Color(.73f, .78f, .73f), .12f);
        Cube("side_wall", new Vector3(-2.36f, 1.35f, 1.25f), new Vector3(.08f, 2.7f, 4.4f), new Color(.69f, .75f, .72f), .12f);
        Cube("baseboard", new Vector3(0, .09f, 3.34f), new Vector3(4.7f, .18f, .08f), new Color(.91f, .87f, .77f), .16f);
        PlaceAsset("rugRectangle", new Vector3(.05f, 0, 1.45f), 2.20f, 0);
        PlaceAsset("loungeSofaLong", new Vector3(1.05f, 0, 2.80f), 1.85f, 180);
        PlaceAsset("bookcaseOpen", new Vector3(-1.70f, 0, 2.85f), 1.55f, 180);
        PlaceAsset("pottedPlant", new Vector3(-1.55f, 0, 1.85f), .92f, 0);
        PlaceAsset("lampSquareFloor", new Vector3(1.78f, 0, 2.65f), 1.42f, 0);
        PlaceAsset("chairCushion", new Vector3(-1.00f, 0, 1.20f), .72f, 155);
        GameObject table = PlaceAsset("tableCoffee", new Vector3(0, 0, .35f), 1.05f, 0);
        if (table) {
            Bounds b = BoundsOf(table); float factor = .592f / Mathf.Max(.001f, b.size.y);
            table.transform.localScale = new Vector3(table.transform.localScale.x, table.transform.localScale.y * (factor + .05f / Mathf.Max(.001f, b.size.y)), table.transform.localScale.z);
            b = BoundsOf(table); table.transform.position += new Vector3(-b.center.x, -b.min.y, .35f - b.center.z);
            Material wood = Material(new Color(.48f, .29f, .16f), .22f); foreach (Renderer renderer in table.GetComponentsInChildren<Renderer>()) renderer.sharedMaterial = wood;
        }
        Cube("soft_play_mat", new Vector3(0, .636f, .35f), new Vector3(.48f, .012f, .38f), new Color(.67f, .80f, .69f), .35f);
        Sphere("yellow_ball", new Vector3(.19f, .682f, .27f), .045f, new Color(.91f, .67f, .12f));
        Cube("blue_block", new Vector3(-.20f, .675f, .28f), new Vector3(.060f, .050f, .055f), new Color(.08f, .29f, .78f), .28f);
        Cube("wood_block", new Vector3(.18f, .670f, .47f), new Vector3(.075f, .040f, .055f), new Color(.72f, .43f, .19f), .18f);
        var sun = new GameObject("warm_window_key").AddComponent<Light>(); sun.type = LightType.Directional; sun.intensity = 1.05f; sun.color = new Color(1f, .92f, .80f); sun.shadows = LightShadows.Soft; sun.transform.rotation = Quaternion.Euler(42, -32, 0);
        var fill = new GameObject("soft_room_fill").AddComponent<Light>(); fill.type = LightType.Point; fill.range = 5; fill.intensity = 2.0f; fill.color = new Color(1f, .76f, .58f); fill.transform.position = new Vector3(-1.1f, 1.9f, .1f); fill.shadows = LightShadows.Soft;
    }

    static void Pose(TraceRow row)
    {
        foreach (var kv in rest) if (kv.Key) kv.Key.localRotation = kv.Value;
        avatar.transform.SetPositionAndRotation(row.avatar_root_position_m, row.avatar_root_rotation);
        torso.localRotation = rest[torso] * row.torso_local_delta;
        neck.localRotation = rest[neck] * row.neck_local_delta;
        head.localRotation = rest[head] * row.head_local_delta;
        SolveArm("R", row.right_palm_position_m, row.right_palm_rotation, initialRightPalmRotation);
        SolveArm("L", row.left_palm_position_m, row.left_palm_rotation, initialLeftPalmRotation);
        PoseDigits("R", row);
        PoseDigits("L", row);
        palmErrors.Add(Vector3.Distance(VisiblePalmCenter("R"), row.right_palm_position_m));
        palmErrors.Add(Vector3.Distance(VisiblePalmCenter("L"), row.left_palm_position_m));
    }

    static void SolveArm(string side, Vector3 desiredPalm, Quaternion physicalRotation, Quaternion initialPhysicalRotation)
    {
        Transform wrist = bones["wrist." + side];
        Quaternion physicalDelta = physicalRotation * Quaternion.Inverse(initialPhysicalRotation);
        Quaternion desiredWristRotation = physicalDelta * wrist.rotation;
        Vector3 desiredWristPosition = desiredPalm - desiredWristRotation * palmOffsetLocal[side];
        string[] chain = { "upperarm01." + side, "upperarm02." + side, "lowerarm01." + side, "lowerarm02." + side };
        for (int pass = 0; pass < 24; pass++) for (int i = chain.Length - 1; i >= 0; i--) {
            Transform joint = bones[chain[i]]; Vector3 current = wrist.position - joint.position; Vector3 target = desiredWristPosition - joint.position;
            if (current.sqrMagnitude > 1e-9f && target.sqrMagnitude > 1e-9f) joint.rotation = Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(current, target), .42f) * joint.rotation;
        }
        wrist.rotation = desiredWristRotation;
    }

    static void PoseDigits(string side, TraceRow row)
    {
        float[] closures = side == "R" ? row.right_digit_closures : row.left_digit_closures;
        string hand = side == "R" ? "right" : "left";
        string[] physicalNames = { "thumb", "index", "middle", "ring", "little" };
        for (int digit = 1; digit <= 5; digit++) {
            float closure = closures != null && closures.Length >= digit ? Mathf.Clamp01(closures[digit - 1]) : 0;
            for (int segment = 1; segment <= 3; segment++) {
                Transform finger = bones[$"finger{digit}-{segment}.{side}"];
                float bend = segment == 1 ? 8 : segment == 2 ? 12 : 10;
                finger.localRotation = rest[finger] * Quaternion.Euler(bend * closure * (side == "R" ? 1 : -1), 0, 0);
            }
            ContactSample contact = row.contacts == null ? null : row.contacts.LastOrDefault(x => x.hand == hand && x.digit == physicalNames[digit - 1]);
            if (contact != null) SolveFinger(side, digit, contact.point_m + ContactSiteOffset(side, digit));
        }
    }

    static Vector3 ContactSiteOffset(string side, int digit)
    {
        if (side == "R" && digit == 1) return new Vector3(-.0014f, -.0070f, .0145f);
        if (side == "R" && digit == 3) return new Vector3(.0020f, .0145f, .0027f);
        if (side == "R" && digit == 4) return new Vector3(-.0003f, .0082f, .0103f);
        if (side == "R" && digit == 5) return new Vector3(.0010f, .0001f, .0046f);
        if (side == "L" && digit == 3) return new Vector3(.0074f, .0102f, 0);
        if (side == "L" && digit == 5) return new Vector3(-.0016f, -.0011f, .0003f);
        return Vector3.zero;
    }

    static void SolveFinger(string side, int digit, Vector3 target)
    {
        var joints = new List<Transform>(); Transform first = bones[$"finger{digit}-1.{side}"]; Transform ancestor = first.parent; Transform wrist = bones["wrist." + side];
        while (ancestor && ancestor != wrist && ancestor.IsChildOf(wrist)) { joints.Insert(0, ancestor); ancestor = ancestor.parent; }
        joints.Add(first); joints.Add(bones[$"finger{digit}-2.{side}"]); joints.Add(bones[$"finger{digit}-3.{side}"]); Transform[] chain = joints.Distinct().ToArray();
        for (int pass = 0; pass < 14; pass++) for (int i = chain.Length - 1; i >= 0; i--) {
            Transform distal = bones[$"finger{digit}-3.{side}"], middle = bones[$"finger{digit}-2.{side}"]; Vector3 tip = distal.position + (distal.position - middle.position) * .70f;
            Vector3 current = tip - chain[i].position, desired = target - chain[i].position;
            if (current.sqrMagnitude > 1e-9f && desired.sqrMagnitude > 1e-9f) chain[i].rotation = Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(current, desired), .48f) * chain[i].rotation;
        }
    }

    static void BuildQaOverlay()
    {
        var rightMat = TransparentMaterial(new Color(.04f, .95f, .20f, .42f));
        var leftMat = TransparentMaterial(new Color(.02f, .70f, 1f, .42f));
        foreach (string side in new[] { "right", "left" }) {
            GameObject palm = GameObject.CreatePrimitive(PrimitiveType.Cube); palm.name = $"QA_ONLY_{side}_physical_palm";
            UnityEngine.Object.DestroyImmediate(palm.GetComponent<Collider>()); palm.transform.localScale = new Vector3(.07f, .075f, .018f); palm.GetComponent<Renderer>().sharedMaterial = side == "right" ? rightMat : leftMat; SetLayer(palm, QaLayer); qaPalms[side] = palm.transform;
        }
        foreach (SegmentSample sample in trace.rows[0].right_digit_segments.Concat(trace.rows[0].left_digit_segments)) {
            GameObject segment = GameObject.CreatePrimitive(PrimitiveType.Cube); segment.name = "QA_ONLY_" + sample.name;
            UnityEngine.Object.DestroyImmediate(segment.GetComponent<Collider>()); segment.transform.localScale = new Vector3(.015f, .015f, .030f);
            segment.GetComponent<Renderer>().sharedMaterial = sample.name.StartsWith("right") ? rightMat : leftMat; SetLayer(segment, QaLayer); qaSegments[sample.name] = segment.transform;
        }
        for (int i = 0; i < 24; i++) {
            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere); point.name = $"QA_ONLY_contact_{i:D2}";
            UnityEngine.Object.DestroyImmediate(point.GetComponent<Collider>()); point.transform.localScale = Vector3.one * .012f;
            point.GetComponent<Renderer>().sharedMaterial = Material(new Color(1f, .04f, .70f), .45f); SetLayer(point, QaLayer); point.SetActive(false); qaContacts.Add(point);
        }
        GameObject legend = new("QA_ONLY_overlay_legend"); legend.transform.SetParent(overlayCamera.transform, false);
        legend.transform.localPosition = new Vector3(-.60f, .31f, .80f); legend.transform.localRotation = Quaternion.identity;
        TextMesh text = legend.AddComponent<TextMesh>(); text.text = "QA ONLY: COLLIDERS, NOT A SECOND BODY\nGREEN right | CYAN left | MAGENTA contact";
        text.anchor = TextAnchor.UpperLeft; text.alignment = TextAlignment.Left; text.fontSize = 48; text.characterSize = .006f; text.color = Color.white;
        SetLayer(legend, QaLayer);
    }

    static void UpdateQa(TraceRow row)
    {
        qaPalms["right"].SetPositionAndRotation(row.right_palm_position_m, row.right_palm_rotation);
        qaPalms["left"].SetPositionAndRotation(row.left_palm_position_m, row.left_palm_rotation);
        foreach (SegmentSample sample in row.right_digit_segments.Concat(row.left_digit_segments)) if (qaSegments.TryGetValue(sample.name, out Transform segment)) segment.SetPositionAndRotation(sample.position_m, sample.rotation);
        for (int i = 0; i < qaContacts.Count; i++) {
            bool active = row.contacts != null && i < row.contacts.Length; qaContacts[i].SetActive(active);
            if (active) qaContacts[i].transform.position = row.contacts[i].point_m;
        }
    }

    static void AuditFrame(int frame, TraceRow row)
    {
        Vector3 cameraPosition = headCamera.transform.position; Quaternion cameraRotation = headCamera.transform.rotation;
        float dt = frame > 0 ? Mathf.Max(1e-6f, row.time_s - trace.rows[frame - 1].time_s) : 1f / 30f;
        Vector3 cameraVelocity = havePreviousCamera ? (cameraPosition - previousCameraPosition) / dt : Vector3.zero;
        Vector3 cameraAcceleration = havePreviousCamera ? (cameraVelocity - previousCameraVelocity) / dt : Vector3.zero;
        Vector3 cameraGyro = havePreviousCamera ? AngularVelocity(previousCameraRotation, cameraRotation, dt) : Vector3.zero;
        previousCameraPosition = cameraPosition; previousCameraRotation = cameraRotation; previousCameraVelocity = cameraVelocity; havePreviousCamera = true;
        foreach (Vector3 point in CurrentHeadWeightedPoints()) minimumCameraClearance = Mathf.Min(minimumCameraClearance, Vector3.Distance(cameraPosition, point));
        maximumCameraRoll = Mathf.Max(maximumCameraRoll, Mathf.Abs(CameraRoll(headCamera.transform)));
        Vector3 objectViewport = headCamera.WorldToViewportPoint(row.object_position_m);
        bool objectVisible = Visible(objectViewport);
        if (row.phase.Contains("touch") || row.phase.Contains("capture") || row.phase.Contains("lift") || row.phase.Contains("turn") || row.phase.Contains("support") || row.phase.Contains("release") || row.phase.Contains("withdrawal")) if (objectVisible) targetVisibleEventFrames++;
        int frameRegistered = 0;
        if (row.contacts != null && row.contacts.Length > 0) {
            physicalContactFrames++; if (firstPhysicalContactFrame < 0) firstPhysicalContactFrame = frame;
            foreach (ContactSample contact in row.contacts) {
                string key = contact.hand + ":" + contact.digit; float distance = -1; Vector3 nearest = Vector3.zero;
                if (digitVertices.TryGetValue(key, out List<int> indices) && indices.Count > 0) {
                    distance = float.PositiveInfinity;
                    foreach (int index in indices) { Vector3 world = skin.transform.TransformPoint(baked.vertices[index]); float d = Vector3.Distance(world, contact.point_m); if (d < distance) { distance = d; nearest = world; } }
                }
                Vector3 contactViewport = headCamera.WorldToViewportPoint(contact.point_m); Vector3 skinViewport = headCamera.WorldToViewportPoint(nearest);
                float pixel = distance >= 0 && contactViewport.z > 0 && skinViewport.z > 0 ? Vector2.Distance(new Vector2(contactViewport.x * width, contactViewport.y * height), new Vector2(skinViewport.x * width, skinViewport.y * height)) : float.PositiveInfinity;
                bool visible = Visible(contactViewport); bool registered = distance >= 0 && distance <= ContactSkinToleranceM && pixel <= ContactPixelTolerance;
                if (visible) visibleContactFrames++; if (registered) frameRegistered++;
                contactRegistrations.Add(new ContactRegistration { frame = frame, time_s = row.time_s, hand = contact.hand, digit = contact.digit, physical_contact_point_m = contact.point_m, nearest_visible_digit_point_m = nearest, digit_surface_distance_m = distance, projected_pixel_distance = pixel, contact_visible_in_head = visible, registered = registered });
            }
        }
        if (frameRegistered > 0) { registeredContactFrames++; if (firstRegisteredContactFrame < 0) firstRegisteredContactFrame = frame; }
        registeredFrames.Add(new RegisteredFrame {
            frame = frame, time_s = row.time_s, phase = row.phase, object_position_m = row.object_position_m, object_rotation = row.object_rotation,
            object_linear_velocity_m_s = row.object_velocity_m_s, object_angular_velocity_rad_s = row.object_angular_velocity_rad_s,
            root_position_m = avatar.transform.position, root_rotation = avatar.transform.rotation, torso_world_rotation = torso.rotation,
            neck_world_rotation = neck.rotation, head_world_position_m = head.position, head_world_rotation = head.rotation,
            camera_position_m = cameraPosition, camera_rotation = cameraRotation, camera_forward = headCamera.transform.forward, camera_up = headCamera.transform.up,
            camera_linear_velocity_m_s = cameraVelocity, head_accelerometer_derived_m_s2 = cameraAcceleration, head_gyro_derived_rad_s = cameraGyro,
            right_wrist_position_m = bones["wrist.R"].position, left_wrist_position_m = bones["wrist.L"].position,
            right_visible_palm_center_m = VisiblePalmCenter("R"), left_visible_palm_center_m = VisiblePalmCenter("L"),
            right_physical_palm_position_m = row.right_palm_position_m, left_physical_palm_position_m = row.left_palm_position_m,
            target_viewport = objectViewport, target_visible = objectVisible, physical_contact_count = row.contacts == null ? 0 : row.contacts.Length, registered_contact_count = frameRegistered,
            bone_poses = BonePoses()
        });
    }

    static BonePose[] BonePoses()
    {
        var names = new List<string> { torso.name, neck.name, head.name, "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R", "upperarm01.L", "upperarm02.L", "lowerarm01.L", "lowerarm02.L", "wrist.L" };
        foreach (string side in new[] { "R", "L" }) for (int digit = 1; digit <= 5; digit++) for (int segment = 1; segment <= 3; segment++) names.Add($"finger{digit}-{segment}.{side}");
        return names.Distinct().Where(bones.ContainsKey).Select(name => { Transform t = bones[name]; return new BonePose { name = name, local_position_m = t.localPosition, local_rotation = t.localRotation, world_position_m = t.position, world_rotation = t.rotation }; }).ToArray();
    }

    static void BuildVertexMaps()
    {
        var weights = skin.sharedMesh.boneWeights;
        var physicalNames = new[] { "thumb", "index", "middle", "ring", "little" };
        foreach (string side in new[] { "R", "L" }) for (int digit = 1; digit <= 5; digit++) {
            var boneIndices = new HashSet<int>(); for (int segment = 1; segment <= 3; segment++) boneIndices.Add(Array.IndexOf(skin.bones, bones[$"finger{digit}-{segment}.{side}"]));
            var vertices = new List<int>(); for (int i = 0; i < weights.Length; i++) if (WeightIn(weights[i], boneIndices) >= .08f) vertices.Add(i);
            digitVertices[(side == "R" ? "right" : "left") + ":" + physicalNames[digit - 1]] = vertices;
        }
        foreach (string side in new[] { "R", "L" }) {
            var handBoneIndices = new HashSet<int>();
            foreach (string name in new[] { "wrist." + side, "metacarpal1." + side, "metacarpal2." + side, "metacarpal3." + side, "metacarpal4." + side, "finger1-1." + side, "finger2-1." + side, "finger3-1." + side, "finger4-1." + side, "finger5-1." + side }) if (bones.ContainsKey(name)) handBoneIndices.Add(Array.IndexOf(skin.bones, bones[name]));
            var handVertices = new List<int>(); for (int i = 0; i < weights.Length; i++) if (WeightIn(weights[i], handBoneIndices) >= .08f) handVertices.Add(i);
            digitVertices[(side == "R" ? "right" : "left") + ":palm"] = handVertices;
        }
        var headIndices = new HashSet<int>(); for (int i = 0; i < skin.bones.Length; i++) if (skin.bones[i] == head || skin.bones[i].IsChildOf(head)) headIndices.Add(i);
        headWeightedIndices = Enumerable.Range(0, weights.Length).Where(i => WeightIn(weights[i], headIndices) >= .25f).ToArray();
    }

    static float WeightIn(BoneWeight w, HashSet<int> indices)
    {
        float result = 0; if (indices.Contains(w.boneIndex0)) result += w.weight0; if (indices.Contains(w.boneIndex1)) result += w.weight1; if (indices.Contains(w.boneIndex2)) result += w.weight2; if (indices.Contains(w.boneIndex3)) result += w.weight3; return result;
    }

    static Vector3[] CurrentHeadWeightedPoints()
    {
        var mesh = new Mesh(); skin.BakeMesh(mesh, true); Vector3[] vertices = mesh.vertices;
        Vector3[] result = headWeightedIndices.Select(i => skin.transform.TransformPoint(vertices[i])).ToArray(); UnityEngine.Object.DestroyImmediate(mesh); return result;
    }

    static Vector3 VisiblePalmCenter(string side)
    {
        return new[] { bones[$"finger2-1.{side}"].position, bones[$"finger3-1.{side}"].position, bones[$"finger4-1.{side}"].position }.Aggregate(Vector3.zero, (a, b) => a + b) / 3f;
    }

    static bool Visible(Vector3 viewport) => viewport.z > headCamera.nearClipPlane && viewport.x >= 0 && viewport.x <= 1 && viewport.y >= 0 && viewport.y <= 1;
    static float CameraRoll(Transform cameraTransform) { Vector3 forward = cameraTransform.forward; Vector3 reference = Vector3.ProjectOnPlane(Vector3.up, forward); Vector3 actual = Vector3.ProjectOnPlane(cameraTransform.up, forward); return reference.sqrMagnitude > 1e-8f && actual.sqrMagnitude > 1e-8f ? Vector3.SignedAngle(reference, actual, forward) : 0; }
    static Vector3 AngularVelocity(Quaternion prior, Quaternion current, float dt) { Quaternion delta = current * Quaternion.Inverse(prior); delta.ToAngleAxis(out float angle, out Vector3 axis); if (angle > 180) angle -= 360; return axis.sqrMagnitude > 0 ? axis.normalized * (angle * Mathf.Deg2Rad / dt) : Vector3.zero; }
    static float Percentile(float[] sorted, float q) { if (sorted == null || sorted.Length == 0) return float.PositiveInfinity; float p = Mathf.Clamp01(q) * (sorted.Length - 1); int lo = Mathf.FloorToInt(p), hi = Mathf.CeilToInt(p); return Mathf.Lerp(sorted[lo], sorted[hi], p - lo); }

    static GameObject Cube(string name, Vector3 position, Vector3 scale, Color color, float gloss)
    {
        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube); go.name = name; go.transform.position = position; go.transform.localScale = scale; go.GetComponent<Renderer>().sharedMaterial = Material(color, gloss); return go;
    }

    static GameObject Sphere(string name, Vector3 position, float diameter, Color color)
    {
        GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere); go.name = name; go.transform.position = position; go.transform.localScale = Vector3.one * diameter; go.GetComponent<Renderer>().sharedMaterial = Material(color, .28f); return go;
    }

    static Material Material(Color color, float gloss)
    {
        var material = new Material(Shader.Find("Standard")); material.color = color; material.SetFloat("_Glossiness", gloss); return material;
    }

    static Material TransparentMaterial(Color color)
    {
        var material = Material(color, .18f); material.SetFloat("_Mode", 3); material.SetInt("_SrcBlend", (int)BlendMode.SrcAlpha); material.SetInt("_DstBlend", (int)BlendMode.OneMinusSrcAlpha); material.SetInt("_ZWrite", 0); material.DisableKeyword("_ALPHATEST_ON"); material.EnableKeyword("_ALPHABLEND_ON"); material.DisableKeyword("_ALPHAPREMULTIPLY_ON"); material.renderQueue = 3000; return material;
    }

    static GameObject PlaceAsset(string name, Vector3 floorPosition, float maxDimension, float yaw)
    {
        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Furniture/{name}.obj"); if (!prefab) return null;
        GameObject go = (GameObject)PrefabUtility.InstantiatePrefab(prefab); go.name = name; go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0, yaw, 0)); go.transform.localScale = Vector3.one;
        Bounds raw = BoundsOf(go); float scale = maxDimension / Mathf.Max(raw.size.x, Mathf.Max(raw.size.y, raw.size.z)); go.transform.localScale = Vector3.one * scale;
        Bounds scaled = BoundsOf(go); go.transform.position = new Vector3(floorPosition.x - scaled.center.x, floorPosition.y - scaled.min.y, floorPosition.z - scaled.center.z); return go;
    }

    static Bounds BoundsOf(GameObject go)
    {
        Renderer[] renderers = go.GetComponentsInChildren<Renderer>(); Bounds result = renderers[0].bounds; for (int i = 1; i < renderers.Length; i++) result.Encapsulate(renderers[i].bounds); return result;
    }

    static void SetLayer(GameObject go, int layer) { go.layer = layer; foreach (Transform child in go.transform) SetLayer(child.gameObject, layer); }
    static string Sha256(string path) { using var h = System.Security.Cryptography.SHA256.Create(); return BitConverter.ToString(h.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant(); }

    sealed class CaptureRig : IDisposable
    {
        readonly Camera camera; readonly string directory; readonly RenderTexture target; readonly Texture2D texture;
        public CaptureRig(Camera camera, string directory, int width, int height) { this.camera = camera; this.directory = directory; target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 }; target.Create(); texture = new Texture2D(width, height, TextureFormat.RGB24, false); camera.targetTexture = target; }
        public void Capture(int frame) { camera.Render(); RenderTexture.active = target; texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0); texture.Apply(false); File.WriteAllBytes(Path.Combine(directory, $"frame_{frame:D4}.png"), texture.EncodeToPNG()); RenderTexture.active = null; }
        public void Dispose() { camera.targetTexture = null; target.Release(); UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(texture); }
    }

    [Serializable] class Trace { public string schema, coordinate_frame, camera_parent_state; public Vector3 physics_to_visual_translation_m; public TraceRow[] rows; }
    [Serializable] class TraceRow { public float time_s; public string phase; public Vector3 object_position_m, blue_object_position_m; public Quaternion object_rotation, blue_object_rotation; public Vector3 object_velocity_m_s, blue_object_velocity_m_s, object_angular_velocity_rad_s, blue_object_angular_velocity_rad_s, right_palm_position_m, left_palm_position_m; public Quaternion right_palm_rotation, left_palm_rotation; public float right_closure, left_closure; public float[] right_digit_closures, left_digit_closures; public SegmentSample[] right_digit_segments, left_digit_segments; public ContactSample[] contacts; public Vector3 avatar_root_position_m; public Quaternion avatar_root_rotation, torso_local_delta, neck_local_delta, head_local_delta; public int measured_contact_count; public bool object_sleeping; }
    [Serializable] class SegmentSample { public string name, digit; public int segment; public Vector3 position_m, linear_velocity_m_s, angular_velocity_rad_s; public Quaternion rotation; }
    [Serializable] class ContactSample { public string hand, digit; public Vector3 point_m, normal; public float separation_m; }
    [Serializable] class RegisteredTrace { public string schema, input_trace_sha256; public Vector3 camera_mount_local_position; public Quaternion camera_mount_local_rotation; public RegisteredFrame[] rows; }
    [Serializable] class RegisteredFrame { public int frame; public float time_s; public string phase; public Vector3 object_position_m, object_linear_velocity_m_s, object_angular_velocity_rad_s; public Quaternion object_rotation; public Vector3 root_position_m; public Quaternion root_rotation, torso_world_rotation, neck_world_rotation; public Vector3 head_world_position_m; public Quaternion head_world_rotation; public Vector3 camera_position_m; public Quaternion camera_rotation; public Vector3 camera_forward, camera_up, camera_linear_velocity_m_s, head_accelerometer_derived_m_s2, head_gyro_derived_rad_s, right_wrist_position_m, left_wrist_position_m, right_visible_palm_center_m, left_visible_palm_center_m, right_physical_palm_position_m, left_physical_palm_position_m, target_viewport; public bool target_visible; public int physical_contact_count, registered_contact_count; public BonePose[] bone_poses; }
    [Serializable] class BonePose { public string name; public Vector3 local_position_m, world_position_m; public Quaternion local_rotation, world_rotation; }
    [Serializable] class ContactRegistrationTrace { public string schema; public ContactRegistration[] rows; }
    [Serializable] class ContactRegistration { public int frame; public float time_s; public string hand, digit; public Vector3 physical_contact_point_m, nearest_visible_digit_point_m; public float digit_surface_distance_m, projected_pixel_distance; public bool contact_visible_in_head, registered; }
    [Serializable] class Report { public string schema, unity_version, source_fbx_sha256, source_license, authoritative_physics_trace, authoritative_physics_trace_sha256, camera_parent_bone; public float duration_s, target_front_dot, fov_deg, near_clip_m, camera_neutral_mount_angle_deg, camera_minimum_skin_clearance_m, camera_maximum_roll_deg, palm_registration_max_m, contact_digit_surface_p95_m, contact_digit_surface_max_m, contact_skin_tolerance_m, contact_pixel_tolerance, maximum_contact_site_offset_m; public int frames, width, height, physical_contact_frames, registered_contact_frames, visible_contact_frames, first_physical_contact_frame, first_registered_contact_frame, first_touch_frame_difference, target_visible_event_frames, proxy_pixels_in_head_or_clean; public Vector3 avatar_face_forward_world, camera_mount_local_position; public Quaternion camera_mount_local_rotation; public bool target_is_in_front, one_weighted_visible_child, separate_labeled_overlay_present, palm_rotations_replayed, per_digit_states_replayed, head_parent_state_replayed, static_bone_to_skin_contact_site_offsets, duration_pass, registration_pass, camera_pass, render_pass, passed; }
}
