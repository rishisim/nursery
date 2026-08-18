using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class UnitySkinRegistrationBuilder
{
    const float Dt = 1f / 240f;
    static readonly string Output = Environment.GetEnvironmentVariable("UNITY_NATIVE_GATE_OUTPUT");
    static readonly string AxisAudit = Environment.GetEnvironmentVariable("UNITY_NATIVE_HEAD_AXIS") ?? "final";
    static readonly string SceneSpecPath = Environment.GetEnvironmentVariable("UNITY_NATIVE_SCENE_SPEC");
    static readonly string ArmCalibrationPath = Environment.GetEnvironmentVariable("UNITY_NATIVE_ARM_CALIBRATION");
    static readonly bool CaptureReplay = Environment.GetEnvironmentVariable("UNITY_NATIVE_CAPTURE") == "1";
    static readonly string ArmCandidateTargets = Environment.GetEnvironmentVariable("UNITY_NATIVE_ARM_TARGETS");
    static GameObject avatar;
    static SkinnedMeshRenderer skin;
    static readonly List<ArticulationBody> bodies = new();
    static readonly Dictionary<string, Transform> bones = new();
    static readonly Dictionary<string, List<SphereCollider>> skinColliders = new();
    static Mesh baked;
    static Vector3 cameraLocalPosition;
    static Quaternion cameraLocalRotation;
    static float neutralMountAngle;
    static float measuredCameraClearance;
    static GameObject target;

    [MenuItem("BabyWorld/Run Unity Episode Trial")]
    public static void RunEpisodeTrial()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_NATIVE_GATE_OUTPUT is required");
        Directory.CreateDirectory(Output); Build(); Physics.simulationMode = SimulationMode.Script; Physics.defaultSolverIterations = 24; Physics.defaultSolverVelocityIterations = 12;
        Camera captureHead = null, captureExternal = null; if (CaptureReplay) { captureHead = BuildHeadCaptureCamera(); captureExternal = CameraAt(new Vector3(1.35f, 1.05f, .95f), new Vector3(.15f, .63f, .18f), 48); Directory.CreateDirectory(Path.Combine(Output, "head_frames")); Directory.CreateDirectory(Path.Combine(Output, "external_frames")); }
        var probe = target.AddComponent<EpisodeContactProbe>(); var rigid = target.GetComponent<Rigidbody>(); var rows = new List<EpisodeRow>(); var jacobianRows = new List<JacobianRow>(); var wristController = new JacobianWristController(avatar.GetComponent<ArticulationBody>(), bodies, bodies.Single(x => x.name == "wrist.R"), jacobianRows); ArmCalibration frozenCalibration = LoadCalibration(); var calibrationDofs = ArmDofs(); var priorDigits = new HashSet<int>(); Vector3 initialPosition = target.transform.position; Vector3 initialShoulder = bones["upperarm01.R"].position, initialWrist = bones["wrist.R"].position; Vector3 initialFingertipCentroid = Enumerable.Range(1, 5).Select(i => bones[$"finger{i}-3.R"].position).Aggregate(Vector3.zero, (a, b) => a + b) / 5; float initialPenetration = FingerTargetPenetration(); int dwell = 0; bool qualified = false; float minimumPrecontactGap = float.PositiveInfinity; bool approachOverlap = false; bool stabilizationContact = false; float settledY = initialPosition.y;
        for (int step = 0; step < 3600; step++) {
            float t = step * Dt;
            bool collisionStopCommand = t < 5f && (priorDigits.Count > 0 || FingerTargetPenetration() > 0); Vector3 waypoint = initialWrist;
            if (t >= 2f) { float behind = t < 3.5f ? .18f : .11f; waypoint = initialPosition + new Vector3(0, .035f, -behind); if (qualified && t >= 7.2f && t < 11f) waypoint.y += .11f; }
            if (frozenCalibration == null) wristController.Step(step, waypoint, collisionStopCommand); else if (t >= 2f && !collisionStopCommand) for (int i = 0; i < calibrationDofs.Count; i++) { float current = GetTarget(calibrationDofs[i]); SetTarget(calibrationDofs[i], Mathf.MoveTowards(current, frozenCalibration.targets_deg[i], .12f)); }
            foreach (var b in bodies.Where(x => x.jointType != ArticulationJointType.FixedJoint)) {
                var d = b.xDrive;
                if (b.name == "head") { d.target = 22; b.xDrive = d; var y = b.yDrive; y.target = -43; b.yDrive = y; var z = b.zDrive; z.target = -10; b.zDrive = z; continue; }
                if (b.name == "neck01") d.target = 0;
                else if (b.name.StartsWith("finger")) { int digit = int.Parse(b.name.Substring(6, 1)); float[] maxima = { 0, -30, 35, 40, 32, 25 }; float desired = t >= 13f ? 0 : maxima[digit]; bool tactileStop = priorDigits.Contains(digit) || DigitPenetration(digit) >= .0025f; if (t < 5f) desired = 0; if (!tactileStop || t >= 13f) d.target = Mathf.MoveTowards(d.target, desired, .18f); }
                else continue;
                b.xDrive = d;
            }
            probe.BeginStep(); Physics.Simulate(Dt); Physics.SyncTransforms(); float currentPenetration = FingerTargetPenetration();
            if (step < 240 && probe.digits.Count > 0) stabilizationContact = true; if (step == 239) settledY = target.transform.position.y;
            if (step >= 1200 && probe.digits.Count >= 3) dwell++; else dwell = 0; if (dwell >= 48) qualified = true;
            if (t >= 2f && t < 5f) { minimumPrecontactGap = Mathf.Min(minimumPrecontactGap, FingerTargetSurfaceGap()); if (FingerTargetPenetration() > 0) approachOverlap = true; }
            var supportObject = GameObject.Find("stage_b_support"); float supportPenetration = Physics.ComputePenetration(target.GetComponent<Collider>(), target.transform.position, target.transform.rotation, supportObject.GetComponent<Collider>(), supportObject.transform.position, supportObject.transform.rotation, out _, out float supportDepth) ? supportDepth : 0;
            rows.Add(new EpisodeRow { step = step, time_s = t, phase = t < 2 ? "scan" : t < 5 ? "reach" : t < 7 ? "grasp" : t < 11 ? "lift_inspect" : t < 13 ? "place" : "release_withdraw", object_position = target.transform.position, object_velocity = rigid.linearVelocity, prior_contact_digits = string.Join(",", priorDigits.OrderBy(x => x)), collision_stop_command = collisionStopCommand, digit_contacts = probe.digits.Count, contact_digits = string.Join(",", probe.digits.OrderBy(x => x)), contact_impulse_n_s = probe.impulse, finger_penetration_m = currentPenetration, support_penetration_m = supportPenetration, qualified = qualified, lift_m = qualified ? target.transform.position.y - settledY : 0 });
            if (CaptureReplay && step % 8 == 0) { int frame = step / 8; CaptureFrame(captureHead, Path.Combine(Output, "head_frames", $"frame_{frame:D4}.png")); CaptureFrame(captureExternal, Path.Combine(Output, "external_frames", $"frame_{frame:D4}.png")); }
            priorDigits = new HashSet<int>(probe.digits);
        }
        float stabilizationDrift = Vector3.Distance(rows[120].object_position, rows[239].object_position); float stabilizationPenetration = rows.Take(240).Max(x => x.support_penetration_m);
        bool validInitialization = initialPenetration <= 0 && !stabilizationContact && stabilizationDrift <= .001f && stabilizationPenetration <= .002f;
        var report = new EpisodeReport { schema = "embodied.unity_native.episode_trial.v1", scene_spec_path = SceneSpecPath, max_digit_contacts = rows.Skip(240).Max(x => x.digit_contacts), qualified = qualified, max_lift_m = rows.Max(x => x.lift_m), initial_finger_penetration_m = initialPenetration, contact_during_stabilization = stabilizationContact, valid_initialization = validInitialization, settled_object_position = rows[239].object_position, final_object_position = target.transform.position, stabilization_drift_m = stabilizationDrift, maximum_support_penetration_m = stabilizationPenetration, free_dynamic_target = true, attachment_or_pose_writes = false, assistance_entries = 0, passed = validInitialization && qualified && rows.Max(x => x.lift_m) >= .08f };
        var stabilization = new StabilizationReceipt { schema = "embodied.unity_native.scene_stabilization.v1", unity_version = Application.unityVersion, physics_backend = "Unity ArticulationBody/PhysX", scene_spec_path = SceneSpecPath, fixed_hz = 240, observed_steps = 240, post_settle_window_steps = 120, target_start_position = initialPosition, target_settled_position = rows[239].object_position, post_settle_drift_m = stabilizationDrift, maximum_support_penetration_m = stabilizationPenetration, passed = stabilizationDrift <= .001f && stabilizationPenetration <= .002f };
        var approach = new DrivenApproachReceipt { schema = "embodied.unity_native.driven_approach.v1", scene_spec_path = SceneSpecPath, initial_shoulder_position_m = initialShoulder, initial_wrist_position_m = initialWrist, initial_fingertip_centroid_m = initialFingertipCentroid, driven_shoulder_position_m = bones["upperarm01.R"].position, driven_wrist_position_m = bones["wrist.R"].position, target_position_m = initialPosition, minimum_fingertip_surface_distance_m = minimumPrecontactGap, overlap_before_closure = approachOverlap, actuator_drives_only = true, target_pose_writes_after_initialization = false, passed = minimumPrecontactGap <= .03f && !approachOverlap };
        File.WriteAllText(Path.Combine(Output, "episode_trial_trace.json"), JsonUtility.ToJson(new EpisodeTrace { rows = rows.ToArray() }, true)); File.WriteAllText(Path.Combine(Output, "jacobian_controller_trace.json"), JsonUtility.ToJson(new JacobianTrace { rows = jacobianRows.ToArray() }, true)); File.WriteAllText(Path.Combine(Output, "episode_trial_report.json"), JsonUtility.ToJson(report, true)); File.WriteAllText(Path.Combine(Output, "scene_stabilization_receipt.json"), JsonUtility.ToJson(stabilization, true)); File.WriteAllText(Path.Combine(Output, "driven_approach_receipt.json"), JsonUtility.ToJson(approach, true)); EditorApplication.Exit(report.passed ? 0 : 2);
    }

    [MenuItem("BabyWorld/Run Right Arm Reach Audit")]
    public static void RunReachAudit()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_NATIVE_GATE_OUTPUT is required");
        Directory.CreateDirectory(Output); Build(); Physics.simulationMode = SimulationMode.Script;
        target.GetComponent<Collider>().enabled = false; target.GetComponent<Rigidbody>().isKinematic = true;
        var rows = new List<ReachAuditRow>(); var cases = new[] { Vector3.zero, new Vector3(40,0,0), new Vector3(-40,0,0), new Vector3(0,40,0), new Vector3(0,-40,0), new Vector3(0,0,40), new Vector3(0,0,-40) };
        foreach (var command in cases) {
            for (int step = 0; step < 360; step++) { foreach (var b in bodies.Where(x => x.jointType != ArticulationJointType.FixedJoint)) { var d = b.xDrive; if (b.name == "upperarm02.R") { d.target = command.x; b.xDrive = d; var y = b.yDrive; y.target = command.y; b.yDrive = y; var z = b.zDrive; z.target = command.z; b.zDrive = z; continue; } else if (b.name == "upperarm01.R") d.target = 50; else if (b.name.StartsWith("lowerarm")) d.target = -30; else if (b.name.StartsWith("finger")) d.target = 0; b.xDrive = d; } Physics.Simulate(Dt); }
            var tips = Enumerable.Range(1, 5).Select(i => bones[$"finger{i}-3.R"].position).ToArray(); var centroid = tips.Aggregate(Vector3.zero, (a, b) => a + b) / tips.Length;
            rows.Add(new ReachAuditRow { shoulder_xyz_target_deg = command, lower_target_deg = -30, wrist_m = bones["wrist.R"].position, fingertip_centroid_m = centroid, target_center_distance_m = Vector3.Distance(centroid, target.transform.position) });
        }
        File.WriteAllText(Path.Combine(Output, "right_arm_reach_audit.json"), JsonUtility.ToJson(new ReachAudit { schema = "embodied.unity_native.right_arm_reach_audit.v1", target_position_m = target.transform.position, target_disabled_calibration_geometry = true, rows = rows.ToArray() }, true)); EditorApplication.Exit(0);
    }

    [MenuItem("BabyWorld/Run Arm Coordinate Descent Calibration")]
    public static void RunArmCalibration()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_NATIVE_GATE_OUTPUT is required"); Directory.CreateDirectory(Output); Build(); Physics.simulationMode = SimulationMode.Script; target.GetComponent<Collider>().enabled = false; target.GetComponent<Rigidbody>().isKinematic = true;
        var refs = ArmDofs(); Vector3 waypoint = target.transform.position + new Vector3(0, .035f, -.11f); var history = new List<CalibrationRow>();
        SettleError(waypoint, 120); foreach (float increment in new[] { 20f, 5f }) foreach (var dof in refs) { float baseline = GetTarget(dof); SetTarget(dof, baseline); float baselineError = SettleError(waypoint, 120); SetTarget(dof, baseline + increment); float plus = SettleError(waypoint, 120); SetTarget(dof, baseline); SettleError(waypoint, 120); SetTarget(dof, baseline - increment); float minus = SettleError(waypoint, 120); float chosen = baseline; if (plus + .001f < baselineError && plus <= minus) chosen = baseline + increment; else if (minus + .001f < baselineError) chosen = baseline - increment; SetTarget(dof, chosen); float chosenError = SettleError(waypoint, 120); history.Add(new CalibrationRow { body = dof.body.name, axis = dof.axis, increment_deg = increment, baseline_error_m = baselineError, plus_error_m = plus, minus_error_m = minus, chosen_target_deg = chosen, chosen_error_m = chosenError }); }
        var result = new ArmCalibration { schema = "embodied.unity_native.arm_coordinate_descent.v1", scene_spec_path = SceneSpecPath, target_disabled_calibration_geometry = true, waypoint_m = waypoint, bodies = refs.Select(x => x.body.name).ToArray(), axes = refs.Select(x => x.axis).ToArray(), targets_deg = refs.Select(GetTarget).ToArray(), final_wrist_m = bones["wrist.R"].position, final_error_m = Vector3.Distance(bones["wrist.R"].position, waypoint), history = history.ToArray() };
        File.WriteAllText(Path.Combine(Output, "arm_coordinate_descent.json"), JsonUtility.ToJson(result, true)); EditorApplication.Exit(0);
    }
    struct ArmDof { public ArticulationBody body; public int axis; }
    static List<ArmDof> ArmDofs() { var result = new List<ArmDof>(); foreach (string name in new[] { "upperarm01.R", "lowerarm01.R", "wrist.R" }) { var b = bodies.Single(x => x.name == name); for (int axis = 0; axis < b.dofCount; axis++) result.Add(new ArmDof { body = b, axis = axis }); } return result; }
    static float GetTarget(ArmDof d) => d.axis == 0 ? d.body.xDrive.target : d.axis == 1 ? d.body.yDrive.target : d.body.zDrive.target;
    static void SetTarget(ArmDof d, float value) { if (d.axis == 0) { var x = d.body.xDrive; x.target = Mathf.Clamp(value, x.lowerLimit, x.upperLimit); d.body.xDrive = x; } else if (d.axis == 1) { var y = d.body.yDrive; y.target = Mathf.Clamp(value, y.lowerLimit, y.upperLimit); d.body.yDrive = y; } else { var z = d.body.zDrive; z.target = Mathf.Clamp(value, z.lowerLimit, z.upperLimit); d.body.zDrive = z; } }
    static float SettleError(Vector3 waypoint, int steps) { for (int i = 0; i < steps; i++) Physics.Simulate(Dt); Physics.SyncTransforms(); return Vector3.Distance(bones["wrist.R"].position, waypoint); }
    static ArmCalibration LoadCalibration() { if (string.IsNullOrWhiteSpace(ArmCalibrationPath)) return null; string json = File.ReadAllText(ArmCalibrationPath); var calibration = JsonUtility.FromJson<ArmCalibration>(json); if (calibration.targets_deg == null || calibration.targets_deg.Length == 0) calibration.targets_deg = JsonUtility.FromJson<FreshCalibrationAggregate>(json).frozen_targets_deg; return calibration; }

    [MenuItem("BabyWorld/Run Fresh Arm Candidate")]
    public static void RunFreshArmCandidate()
    {
        if (string.IsNullOrWhiteSpace(Output) || string.IsNullOrWhiteSpace(ArmCandidateTargets)) throw new Exception("output and candidate targets are required"); Directory.CreateDirectory(Output); Build(); Physics.simulationMode = SimulationMode.Script; target.GetComponent<Collider>().enabled = false; target.GetComponent<Rigidbody>().isKinematic = true;
        var refs = ArmDofs(); var values = ArmCandidateTargets.Split(',').Select(x => float.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray(); if (values.Length != refs.Count) throw new Exception("candidate must have seven values"); string initialHash = ArmStateHash(refs); for (int i = 0; i < refs.Count; i++) SetTarget(refs[i], values[i]); Vector3 waypoint = target.transform.position + new Vector3(0, .035f, -.11f); float error = SettleError(waypoint, 480); string finalHash = ArmStateHash(refs);
        var receipt = new FreshCandidate { schema = "embodied.unity_native.fresh_arm_candidate.v1", scene_spec_path = SceneSpecPath, fixed_steps = 480, fixed_hz = 240, target_disabled_calibration_geometry = true, initial_state_hash = initialHash, final_state_hash = finalHash, targets_deg = values, waypoint_m = waypoint, final_wrist_m = bones["wrist.R"].position, final_error_m = error };
        File.WriteAllText(Path.Combine(Output, "fresh_candidate.json"), JsonUtility.ToJson(receipt, true)); EditorApplication.Exit(0);
    }
    static string ArmStateHash(List<ArmDof> refs) { var s = new StringBuilder(); foreach (var b in bodies.OrderBy(x => x.index)) { s.Append(b.name).Append('|').Append(b.transform.position.ToString("R")).Append('|').Append(b.transform.rotation.ToString("R")).Append('|'); for (int i = 0; i < b.jointPosition.dofCount; i++) s.Append(b.jointPosition[i].ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(','); for (int i = 0; i < b.jointVelocity.dofCount; i++) s.Append(b.jointVelocity[i].ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(','); } foreach (var d in refs) s.Append(GetTarget(d).ToString("R", System.Globalization.CultureInfo.InvariantCulture)).Append(','); using var sha = SHA256.Create(); return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(s.ToString()))).Replace("-", "").ToLowerInvariant(); }

    [MenuItem("BabyWorld/Run Unity Skin Registration")]
    public static void Run()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_NATIVE_GATE_OUTPUT is required");
        Directory.CreateDirectory(Output); Build();
        Physics.simulationMode = SimulationMode.Script; Physics.defaultSolverIterations = 24; Physics.defaultSolverVelocityIterations = 12;
        var restLengths = SegmentLengths(); var rows = new List<SkinRow>();
        for (int step = 0; step < 480; step++) {
            float amount = Mathf.SmoothStep(0, 1, Mathf.InverseLerp(120, 360, step));
            foreach (var b in bodies.Where(x => x.jointType != ArticulationJointType.FixedJoint)) {
                var d = b.xDrive;
                if (b.name == "head") { float audit = AxisAudit == "baseline" ? 0 : 12f * amount; d.target = AxisAudit == "x" ? audit : AxisAudit == "final" ? 22f * amount : 0; d.stiffness = 1000; d.damping = 50; d.forceLimit = 100; b.xDrive = d; var yaw = b.yDrive; yaw.target = AxisAudit == "y" ? audit : AxisAudit == "final" ? -43f * amount : 0; yaw.stiffness = 1000; yaw.damping = 50; yaw.forceLimit = 100; b.yDrive = yaw; var pitch = b.zDrive; pitch.target = AxisAudit == "z" ? audit : AxisAudit == "final" ? -10f * amount : 0; pitch.stiffness = 1000; pitch.damping = 50; pitch.forceLimit = 100; b.zDrive = pitch; continue; }
                else if (b.name == "neck01") d.target = 0;
                else if (b.name.StartsWith("upperarm")) d.target = 50f * amount;
                else if (b.name.StartsWith("lowerarm")) d.target = -45f * amount;
                else d.target = (b.name.Contains("finger1-") ? -18f : 22f) * amount;
                b.xDrive = d;
            }
            Physics.Simulate(Dt); Physics.SyncTransforms(); skin.BakeMesh(baked, true); baked.RecalculateBounds();
            if (step % 8 == 0) {
                var registration = ColliderSkinErrors();
                var head = bones["head"]; var headBody = bodies.Single(x => x.name == "head");
                rows.Add(new SkinRow { step = step, mesh_bounds = baked.bounds, segment_length_max_error_m = SegmentError(restLengths), collider_skin_errors_m = registration, head_joint_position = Values(headBody.jointPosition), head_drive_targets_deg = new[] { headBody.xDrive.target, headBody.yDrive.target, headBody.zDrive.target }, camera_position = head.TransformPoint(cameraLocalPosition), camera_forward = head.TransformDirection(cameraLocalRotation * Vector3.forward), finite = Finite(baked.bounds) });
            }
        }
        RenderExternal("registration_clean.png", false);
        RenderExternal("registration_colliders_QA_ONLY.png", true);
        var fovMetrics = new List<FovMetric>(); foreach (int fov in new[] { 60, 68, 75 }) fovMetrics.Add(RenderHead($"pov_{fov}.png", fov));
        measuredCameraClearance = FinalCameraSkinClearance();
        var allRegistration = rows.SelectMany(x => x.collider_skin_errors_m).OrderBy(x => x).ToArray();
        var report = new SkinReport {
            schema = "embodied.unity_native.skin_registration.v1", unity_version = Application.unityVersion,
            actual_weighted_skin = true, articulation_is_only_pose_state = true, independently_advanced_animation = false,
            selected_articulation_bodies = bodies.Select(x => x.name).ToArray(), fixed_physics_hz = 240,
            segment_length_max_error_m = rows.Max(x => x.segment_length_max_error_m),
            collider_skin_median_m = Quantile(allRegistration, .5f), collider_skin_p95_m = Quantile(allRegistration, .95f), collider_skin_max_m = allRegistration.Max(),
            mesh_bounds_ratio_max = rows.Max(x => MaxRatio(rows[0].mesh_bounds.size, x.mesh_bounds.size)),
            all_finite = rows.All(x => x.finite), qa_colliders_in_clean_or_pov = false,
            camera_mount_local_position = cameraLocalPosition,
            camera_mount_local_euler = cameraLocalRotation.eulerAngles, camera_clearance_m = measuredCameraClearance,
            neutral_face_forward_angle_deg = neutralMountAngle,
            excluded_adjacent_collision_pairs = ExcludedPairs().ToArray(), retained_distal_collision_pairs = RetainedPairs().ToArray(),
            target_position = target.transform.position, wrist_position = bones["wrist.R"].position,
            fingertip_positions = Enumerable.Range(1, 5).Select(i => bones[$"finger{i}-3.R"].position).ToArray(),
            final_camera_forward = bones["head"].TransformDirection(cameraLocalRotation * Vector3.forward),
            final_camera_up = bones["head"].TransformDirection(cameraLocalRotation * Vector3.up),
            camera_to_target_direction = (target.transform.position - bones["head"].TransformPoint(cameraLocalPosition)).normalized,
            head_axis_audit = AxisAudit, head_joint_position = Values(bodies.Single(x => x.name == "head").jointPosition), head_drive_targets_deg = new[] { bodies.Single(x => x.name == "head").xDrive.target, bodies.Single(x => x.name == "head").yDrive.target, bodies.Single(x => x.name == "head").zDrive.target },
            neck_joint_position = Values(bodies.Single(x => x.name == "neck01").jointPosition), neck_drive_target_deg = bodies.Single(x => x.name == "neck01").xDrive.target,
            torso_joint_position = Values(bodies.Single(x => x.name == "spine03").jointPosition), fov_metrics = fovMetrics.ToArray(), frozen_fov_deg = 68,
            camera_roll_deg = CameraRoll(), camera_skin_clearance_method = "minimum Euclidean distance from final optical origin and 75-degree near-plane center/corners to final baked vertices with >=0.25 head-bone weight",
            passed = rows.All(x => x.finite) && allRegistration.Max() <= .0075f && rows.Max(x => MaxRatio(rows[0].mesh_bounds.size, x.mesh_bounds.size)) <= 1.15f && fovMetrics.All(x => x.target_visible && x.visible_fingertips >= 3) && Mathf.Abs(CameraRoll()) <= 8f && FinalCameraSkinClearance() >= .02f
        };
        File.WriteAllText(Path.Combine(Output, "skin_registration_trace.json"), JsonUtility.ToJson(new SkinTrace { rows = rows.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "skin_registration_report.json"), JsonUtility.ToJson(report, true));
        EditorApplication.Exit(report.passed ? 0 : 2);
    }

    static void Build()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        bodies.Clear(); bones.Clear(); skinColliders.Clear();
        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx")); avatar.name = "CC0_Weighted_MPFB_Child"; avatar.transform.localScale = Vector3.one * 1.9f;
        skin = avatar.GetComponentsInChildren<SkinnedMeshRenderer>().Single(); skin.updateWhenOffscreen = true; skin.localBounds = new Bounds(Vector3.zero, Vector3.one * 4);
        foreach (var t in avatar.GetComponentsInChildren<Transform>(true)) bones[t.name] = t;
        var root = avatar.AddComponent<ArticulationBody>(); root.immovable = true; bodies.Add(root);
        string[] fixedNames = { "spine01", "spine02", "spine03", "neck", "head", "upperarm01.R", "upperarm02.R", "lowerarm01.R", "lowerarm02.R", "wrist.R" };
        var headAncestors = new List<Transform>(); for (var p = bones["head"].parent; p && p != avatar.transform; p = p.parent) headAncestors.Add(p); headAncestors.Reverse();
        foreach (var p in headAncestors) if (!p.GetComponent<ArticulationBody>()) AddBody(p.name, false);
        var neckBody = bodies.Single(x => x.name == "neck01"); neckBody.jointType = ArticulationJointType.RevoluteJoint; neckBody.twistLock = ArticulationDofLock.LimitedMotion; var neckDrive = neckBody.xDrive; neckDrive.lowerLimit = -40; neckDrive.upperLimit = 40; neckDrive.stiffness = 1000; neckDrive.damping = 50; neckDrive.forceLimit = 100; neckBody.xDrive = neckDrive;
        foreach (var name in fixedNames) if (bones.ContainsKey(name)) AddBody(name, name == "head" || name.StartsWith("upperarm") || name.StartsWith("lowerarm") || name == "wrist.R");
        var shoulder = bodies.Single(x => x.name == "upperarm01.R"); MakeSpherical(shoulder);
        bodies.Single(x => x.name == "upperarm02.R").jointType = ArticulationJointType.FixedJoint;
        bodies.Single(x => x.name == "lowerarm02.R").jointType = ArticulationJointType.FixedJoint;
        var wristBody = bodies.Single(x => x.name == "wrist.R"); MakeSpherical(wristBody);
        var headBody = bodies.Single(x => x.name == "head"); headBody.jointType = ArticulationJointType.SphericalJoint; headBody.twistLock = ArticulationDofLock.LimitedMotion; headBody.swingYLock = ArticulationDofLock.LimitedMotion; headBody.swingZLock = ArticulationDofLock.LimitedMotion;
        var headY = headBody.yDrive; headY.lowerLimit = -50; headY.upperLimit = 35; headY.stiffness = 12; headY.damping = 2.5f; headY.forceLimit = .8f; headBody.yDrive = headY;
        var headZ = headBody.zDrive; headZ.lowerLimit = -45; headZ.upperLimit = 45; headZ.stiffness = 1000; headZ.damping = 50; headZ.forceLimit = 100; headBody.zDrive = headZ;
        for (int digit = 1; digit <= 5; digit++) for (int part = 1; part <= 3; part++) AddBody($"finger{digit}-{part}.R", true);
        baked = new Mesh { name = "ActualWeightedSkinRegistrationFrame" };
        FitSkinColliders(); ConfigureCollisionMatrix(); MeasureCameraMount();
        var floor = GameObject.CreatePrimitive(PrimitiveType.Cube); floor.name = "room_floor"; floor.transform.position = new Vector3(0, -.04f, -.8f); floor.transform.localScale = new Vector3(3, .08f, 3);
        var wall = GameObject.CreatePrimitive(PrimitiveType.Cube); wall.name = "room_wall"; wall.transform.position = new Vector3(0, 1.25f, -2.25f); wall.transform.localScale = new Vector3(3, 2.5f, .08f);
        SceneInput scene = !string.IsNullOrWhiteSpace(SceneSpecPath) ? JsonUtility.FromJson<SceneInput>(File.ReadAllText(SceneSpecPath)) : null;
        var targetPosition = scene != null ? V(scene.target.transform.position_m) : new Vector3(.35f, .66f, .30f); var targetScale = scene != null ? V(scene.target.transform.scale_m) : Vector3.one * .045f;
        target = GameObject.CreatePrimitive(PrimitiveType.Cube); target.name = "stage_b_free_target"; target.transform.localScale = targetScale; target.transform.position = targetPosition;
        var targetMaterial = new Material(Shader.Find("Standard")); targetMaterial.color = new Color(.9f, .04f, .02f); target.GetComponent<Renderer>().sharedMaterial = targetMaterial;
        var targetBody = target.AddComponent<Rigidbody>(); targetBody.mass = .045f; targetBody.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic; target.GetComponent<Collider>().material = HighFriction();
        var supportSpec = scene?.instances?.FirstOrDefault(x => x.persistent_id == "primary_support"); var support = GameObject.CreatePrimitive(PrimitiveType.Cube); support.name = "stage_b_support";
        if (supportSpec != null) { var envelopeScale = V(supportSpec.transform.scale_m); var envelopePosition = V(supportSpec.transform.position_m); const float topThickness = .04f; support.transform.localScale = new Vector3(envelopeScale.x, topThickness, envelopeScale.z); support.transform.position = new Vector3(envelopePosition.x, envelopePosition.y + envelopeScale.y * .5f - topThickness * .5f, envelopePosition.z); }
        else { support.transform.localScale = new Vector3(.5f, .04f, .45f); support.transform.position = new Vector3(target.transform.position.x, target.transform.position.y - .0425f, target.transform.position.z); }
        var light = new GameObject("key").AddComponent<Light>(); light.type = LightType.Directional; light.intensity = 1.1f; light.transform.rotation = Quaternion.Euler(45, -30, 0); light.shadows = LightShadows.Soft;
    }

    static void AddBody(string name, bool revolute)
    {
        if (!bones.TryGetValue(name, out var t)) throw new Exception("Missing weighted bone " + name);
        if (t.GetComponent<ArticulationBody>()) return;
        var body = t.gameObject.AddComponent<ArticulationBody>(); bodies.Add(body);
        if (!revolute) { body.jointType = ArticulationJointType.FixedJoint; return; }
        body.jointType = ArticulationJointType.RevoluteJoint; body.twistLock = ArticulationDofLock.LimitedMotion;
        var d = body.xDrive; d.lowerLimit = -60; d.upperLimit = 60; bool proximal = name.StartsWith("upperarm") || name.StartsWith("lowerarm") || name == "wrist.R"; d.stiffness = proximal ? 200 : 40; d.damping = proximal ? 12 : 4; d.forceLimit = proximal ? 20 : 3; body.xDrive = d;
    }
    static void ConfigureShoulderDrive(ArticulationBody body, char axis) { var d = axis == 'y' ? body.yDrive : body.zDrive; d.lowerLimit = -60; d.upperLimit = 60; d.stiffness = 200; d.damping = 12; d.forceLimit = 20; if (axis == 'y') body.yDrive = d; else body.zDrive = d; }
    static void MakeSpherical(ArticulationBody body) { body.jointType = ArticulationJointType.SphericalJoint; body.twistLock = ArticulationDofLock.LimitedMotion; body.swingYLock = ArticulationDofLock.LimitedMotion; body.swingZLock = ArticulationDofLock.LimitedMotion; ConfigureShoulderDrive(body, 'y'); ConfigureShoulderDrive(body, 'z'); }

    static void FitSkinColliders() {
        var mesh = skin.sharedMesh; var weights = mesh.boneWeights; var vertices = mesh.vertices;
        foreach (var body in bodies.Where(x => x.name.StartsWith("finger"))) {
            int boneIndex = Array.IndexOf(skin.bones, body.transform); var local = new List<Vector3>();
            for (int i = 0; i < weights.Length; i++) { var w = weights[i]; float weight = 0; if (w.boneIndex0 == boneIndex) weight += w.weight0; if (w.boneIndex1 == boneIndex) weight += w.weight1; if (w.boneIndex2 == boneIndex) weight += w.weight2; if (w.boneIndex3 == boneIndex) weight += w.weight3; if (weight >= .60f) local.Add(body.transform.InverseTransformPoint(skin.transform.TransformPoint(vertices[i]))); }
            if (local.Count == 0) throw new Exception("No dominant weighted vertices for " + body.name);
            var bounds = new Bounds(local[0], Vector3.zero); foreach (var p in local) bounds.Encapsulate(p); int axis = bounds.size.x >= bounds.size.y && bounds.size.x >= bounds.size.z ? 0 : bounds.size.y >= bounds.size.z ? 1 : 2;
            var ordered = local.OrderBy(p => p[axis]).ToArray(); int cut1 = ordered.Length / 3, cut2 = ordered.Length * 2 / 3; var groups = new[] { ordered.Take(cut1).ToArray(), ordered.Skip(cut1).Take(cut2 - cut1).ToArray(), ordered.Skip(cut2).ToArray() }; var fitted = new List<SphereCollider>();
            foreach (var group in groups.Where(g => g.Length > 0)) { var center = group.Aggregate(Vector3.zero, (a, b) => a + b) / group.Length; var radii = group.Select(x => Vector3.Distance(x, center)).OrderBy(x => x).ToArray(); var c = body.gameObject.AddComponent<SphereCollider>(); c.center = center; c.radius = (radii.First() + radii.Last()) * .5f; c.material = HighFriction(); fitted.Add(c); }
            skinColliders[body.name] = fitted;
        }
    }
    static void ConfigureCollisionMatrix() { var rows = skinColliders.ToArray(); for (int i = 0; i < rows.Length; i++) for (int j = i + 1; j < rows.Length; j++) { var a = rows[i]; var b = rows[j]; if (Adjacent(a.Key, b.Key)) foreach (var ac in a.Value) foreach (var bc in b.Value) Physics.IgnoreCollision(ac, bc, true); } }
    static bool Adjacent(string a, string b) { var aa = a.Split(new[] {'-', '.'}); var bb = b.Split(new[] {'-', '.'}); return aa[0] == bb[0] && Math.Abs(int.Parse(aa[1]) - int.Parse(bb[1])) <= 1; }
    static IEnumerable<string> ExcludedPairs() { var rows = skinColliders.Keys.OrderBy(x => x).ToArray(); for (int i = 0; i < rows.Length; i++) for (int j = i + 1; j < rows.Length; j++) if (Adjacent(rows[i], rows[j])) yield return rows[i] + "|" + rows[j]; }
    static IEnumerable<string> RetainedPairs() { var rows = skinColliders.Keys.OrderBy(x => x).ToArray(); for (int i = 0; i < rows.Length; i++) for (int j = i + 1; j < rows.Length; j++) if (!Adjacent(rows[i], rows[j]) && rows[i].EndsWith("3.R") && rows[j].EndsWith("3.R")) yield return rows[i] + "|" + rows[j]; }
    static void MeasureCameraMount() { var head = bones["head"]; var mesh = skin.sharedMesh; var weights = mesh.boneWeights; int headIndex = Array.IndexOf(skin.bones, head); var points = new List<Vector3>(); for (int i = 0; i < weights.Length; i++) { var w = weights[i]; float weight = 0; if (w.boneIndex0 == headIndex) weight += w.weight0; if (w.boneIndex1 == headIndex) weight += w.weight1; if (w.boneIndex2 == headIndex) weight += w.weight2; if (w.boneIndex3 == headIndex) weight += w.weight3; if (weight >= .25f) points.Add(skin.transform.TransformPoint(mesh.vertices[i])); } var bounds = new Bounds(points[0], Vector3.zero); foreach (var p in points) bounds.Encapsulate(p); var world = new Vector3(bounds.center.x, bounds.center.y, bounds.max.z + .032f); cameraLocalPosition = head.InverseTransformPoint(world); cameraLocalRotation = Quaternion.Inverse(head.rotation) * Quaternion.LookRotation(Vector3.forward, Vector3.up); neutralMountAngle = Vector3.Angle(head.TransformDirection(cameraLocalRotation * Vector3.forward), Vector3.forward); measuredCameraClearance = points.Min(p => Vector3.Distance(world, p)); }
    static float[] ColliderSkinErrors() { var mesh = skin.sharedMesh; var weights = mesh.boneWeights; var vertices = baked.vertices; var errors = new List<float>(); foreach (var row in skinColliders) { var body = bodies.Single(x => x.name == row.Key); int boneIndex = Array.IndexOf(skin.bones, body.transform); for (int i = 0; i < weights.Length; i++) { var w = weights[i]; float weight = 0; if (w.boneIndex0 == boneIndex) weight += w.weight0; if (w.boneIndex1 == boneIndex) weight += w.weight1; if (w.boneIndex2 == boneIndex) weight += w.weight2; if (w.boneIndex3 == boneIndex) weight += w.weight3; if (weight >= .60f) { Vector3 vertex = skin.transform.TransformPoint(vertices[i]); errors.Add(row.Value.Min(c => { Vector3 center = c.transform.TransformPoint(c.center); float radius = c.radius * Mathf.Max(c.transform.lossyScale.x, Mathf.Max(c.transform.lossyScale.y, c.transform.lossyScale.z)); return Mathf.Abs(Vector3.Distance(center, vertex) - radius); })); } } } return errors.ToArray(); }
    static float Quantile(float[] values, float q) { if (values.Length == 0) return float.PositiveInfinity; var sorted = values.OrderBy(x => x).ToArray(); return sorted[Mathf.Clamp(Mathf.RoundToInt((sorted.Length - 1) * q), 0, sorted.Length - 1)]; }
    static float[] Values(ArticulationReducedSpace value) { var result = new float[value.dofCount]; for (int i = 0; i < value.dofCount; i++) result[i] = value[i]; return result; }
    static Vector3 V(float[] value) { if (value == null || value.Length != 3) throw new Exception("SceneSpec vector must contain three values"); return new Vector3(value[0], value[1], value[2]); }
    static PhysicsMaterial HighFriction() => new PhysicsMaterial { dynamicFriction = 1.0f, staticFriction = 1.0f, frictionCombine = PhysicsMaterialCombine.Maximum, bounceCombine = PhysicsMaterialCombine.Minimum };
    static float FingerTargetSurfaceGap() { var tc = target.GetComponent<Collider>(); return skinColliders.Values.SelectMany(x => x).Min(c => Vector3.Distance(c.ClosestPoint(tc.bounds.center), tc.ClosestPoint(c.bounds.center))); }
    static float FingerTargetPenetration() { var tc = target.GetComponent<Collider>(); float maximum = 0; foreach (var c in skinColliders.Values.SelectMany(x => x)) if (Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, tc, tc.transform.position, tc.transform.rotation, out _, out float depth)) maximum = Mathf.Max(maximum, depth); return maximum; }
    static float DigitPenetration(int digit) { var tc = target.GetComponent<Collider>(); float maximum = 0; foreach (var row in skinColliders.Where(x => x.Key.StartsWith($"finger{digit}-"))) foreach (var c in row.Value) if (Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, tc, tc.transform.position, tc.transform.rotation, out _, out float depth)) maximum = Mathf.Max(maximum, depth); return maximum; }

    static Dictionary<string, float> SegmentLengths() => bodies.Where(x => x.transform.parent).ToDictionary(x => x.name, x => Vector3.Distance(x.transform.position, x.transform.parent.position));
    static float SegmentError(Dictionary<string, float> rest) => bodies.Where(x => x.transform.parent && rest.ContainsKey(x.name)).Max(x => Mathf.Abs(Vector3.Distance(x.transform.position, x.transform.parent.position) - rest[x.name]));
    static bool Finite(Bounds b) => float.IsFinite(b.center.x) && float.IsFinite(b.center.y) && float.IsFinite(b.center.z) && float.IsFinite(b.size.x) && float.IsFinite(b.size.y) && float.IsFinite(b.size.z);
    static float MaxRatio(Vector3 a, Vector3 b) => Mathf.Max(b.x / Mathf.Max(a.x, 1e-5f), b.y / Mathf.Max(a.y, 1e-5f), b.z / Mathf.Max(a.z, 1e-5f));
    static Camera CameraAt(Vector3 p, Vector3 target, float fov) { var c = new GameObject("capture_camera").AddComponent<Camera>(); c.transform.SetPositionAndRotation(p, Quaternion.LookRotation(target - p, Vector3.up)); c.fieldOfView = fov; c.nearClipPlane = .03f; c.farClipPlane = 10; return c; }
    static Camera BuildHeadCaptureCamera() { var c = new GameObject("rigid_head_failure_capture").AddComponent<Camera>(); c.transform.SetParent(bones["head"], false); c.transform.localPosition = cameraLocalPosition; c.transform.localRotation = cameraLocalRotation; c.fieldOfView = 68; c.nearClipPlane = .03f; c.farClipPlane = 10; c.cullingMask &= ~(1 << 30); return c; }
    static void CaptureFrame(Camera c, string path) { var rt = new RenderTexture(640, 360, 24); var tex = new Texture2D(640, 360, TextureFormat.RGB24, false); c.targetTexture = rt; c.Render(); RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, 640, 360), 0, 0); tex.Apply(); File.WriteAllBytes(path, tex.EncodeToPNG()); c.targetTexture = null; RenderTexture.active = null; UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(tex); }
    static void RenderExternal(string filename, bool overlay) { if (overlay) BuildWorldOverlay(); var c = CameraAt(new Vector3(1.45f, 1.05f, .95f), new Vector3(0, .68f, 0), 46); c.cullingMask = overlay ? -1 : ~(1 << 30); Render(c, filename); }
    static void BuildWorldOverlay() {
        foreach (var source in avatar.GetComponentsInChildren<SphereCollider>(true)) {
            var q = GameObject.CreatePrimitive(PrimitiveType.Cube); q.name = "QA_ONLY_COLLIDER_BOUNDS_" + source.transform.parent.name; q.layer = 30; UnityEngine.Object.DestroyImmediate(q.GetComponent<Collider>());
            q.transform.position = source.bounds.center; q.transform.localScale = source.bounds.size;
            var m = new Material(Shader.Find("Standard")); m.color = new Color(.05f, .9f, .35f, .5f); q.GetComponent<Renderer>().sharedMaterial = m;
        }
    }
    static FovMetric RenderHead(string filename, int fov) {
        var head = bones["head"]; var c = new GameObject("rigid_head_camera").AddComponent<Camera>(); c.transform.SetParent(head, false); c.transform.localPosition = cameraLocalPosition; c.transform.localRotation = cameraLocalRotation;
        c.fieldOfView = fov; c.nearClipPlane = .03f; c.farClipPlane = 10; c.cullingMask &= ~(1 << 30);
        var center = c.WorldToViewportPoint(target.GetComponent<Renderer>().bounds.center); var fingertips = Enumerable.Range(1, 5).Select(i => c.WorldToViewportPoint(bones[$"finger{i}-3.R"].position)).ToArray(); var wrist = c.WorldToViewportPoint(bones["wrist.R"].position);
        var metric = new FovMetric { fov_deg = fov, target_center_viewport = center, wrist_viewport = wrist, fingertip_viewports = fingertips, target_visible = Visible(center), wrist_visible = Visible(wrist), visible_fingertips = fingertips.Count(Visible) };
        Render(c, filename); return metric;
    }
    static bool Visible(Vector3 p) => p.z >= .03f && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
    static float CameraRoll() { var head = bones["head"]; Vector3 forward = head.TransformDirection(cameraLocalRotation * Vector3.forward); Vector3 up = head.TransformDirection(cameraLocalRotation * Vector3.up); return Vector3.SignedAngle(Vector3.ProjectOnPlane(Vector3.up, forward), Vector3.ProjectOnPlane(up, forward), forward); }
    static float FinalCameraSkinClearance() {
        var head = bones["head"]; Vector3 origin = head.TransformPoint(cameraLocalPosition); Quaternion rotation = head.rotation * cameraLocalRotation; float near = .03f, halfH = Mathf.Tan(75f * .5f * Mathf.Deg2Rad) * near, halfW = halfH * (16f / 9f);
        var envelope = new List<Vector3> { origin, origin + rotation * new Vector3(0, 0, near), origin + rotation * new Vector3(halfW, halfH, near), origin + rotation * new Vector3(-halfW, halfH, near), origin + rotation * new Vector3(halfW, -halfH, near), origin + rotation * new Vector3(-halfW, -halfH, near) };
        var mesh = skin.sharedMesh; var weights = mesh.boneWeights; int headIndex = Array.IndexOf(skin.bones, head); var points = new List<Vector3>(); for (int i = 0; i < weights.Length; i++) { var w = weights[i]; float weight = 0; if (w.boneIndex0 == headIndex) weight += w.weight0; if (w.boneIndex1 == headIndex) weight += w.weight1; if (w.boneIndex2 == headIndex) weight += w.weight2; if (w.boneIndex3 == headIndex) weight += w.weight3; if (weight >= .25f) points.Add(skin.transform.TransformPoint(baked.vertices[i])); }
        return envelope.Min(e => points.Min(p => Vector3.Distance(e, p)));
    }
    static void Render(Camera c, string filename) { var rt = new RenderTexture(960, 540, 24); var tex = new Texture2D(960, 540, TextureFormat.RGB24, false); c.targetTexture = rt; c.Render(); RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, 960, 540), 0, 0); tex.Apply(); File.WriteAllBytes(Path.Combine(Output, filename), tex.EncodeToPNG()); RenderTexture.active = null; UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(tex); UnityEngine.Object.DestroyImmediate(c.gameObject); }

    sealed class JacobianWristController
    {
        readonly ArticulationBody root, wrist; readonly List<JacobianRow> trace; readonly List<int> starts = new(); readonly List<DofRef> selected = new(); ArticulationJacobian jacobian; const float Damping = .08f;
        struct DofRef { public ArticulationBody body; public int axis, column; }
        public JacobianWristController(ArticulationBody rootBody, List<ArticulationBody> allBodies, ArticulationBody wristBody, List<JacobianRow> output) {
            root = rootBody; wrist = wristBody; trace = output; int columns = root.GetDofStartIndices(starts); jacobian = new ArticulationJacobian(allBodies.Count * 6, columns);
            foreach (var body in allBodies.Where(x => x.name == "upperarm01.R" || x.name == "lowerarm01.R" || x.name == "wrist.R")) for (int axis = 0; axis < body.dofCount; axis++) selected.Add(new DofRef { body = body, axis = axis, column = starts[body.index] + axis });
        }
        public void Step(int step, Vector3 waypoint, bool collisionStop) {
            root.GetDenseJacobian(ref jacobian); Vector3 error = waypoint - wrist.transform.position; int row = wrist.index * 6;
            var normal = Matrix4x4.identity; for (int r = 0; r < 3; r++) for (int s = 0; s < 3; s++) { float value = r == s ? Damping * Damping : 0; foreach (var dof in selected) value += jacobian[row + r, dof.column] * jacobian[row + s, dof.column]; normal[r, s] = value; }
            Vector3 solved = normal.inverse.MultiplyVector(error * 5f);
            if (!collisionStop) foreach (var dof in selected) { float radiansPerSecond = jacobian[row, dof.column] * solved.x + jacobian[row + 1, dof.column] * solved.y + jacobian[row + 2, dof.column] * solved.z; ApplyIncrement(dof, Mathf.Clamp(radiansPerSecond * Dt * Mathf.Rad2Deg, -.35f, .35f)); }
            if (step % 8 == 0) trace.Add(new JacobianRow { step = step, rows = jacobian.rows, columns = jacobian.columns, selected_columns = selected.Select(x => x.column).ToArray(), damping = Damping, error_m = error.magnitude, normal_determinant = normal.determinant, wrist_m = wrist.transform.position, waypoint_m = waypoint, collision_stop = collisionStop });
        }
        static void ApplyIncrement(DofRef dof, float delta) { if (dof.axis == 0) { var d = dof.body.xDrive; d.target = Mathf.Clamp(d.target + delta, d.lowerLimit, d.upperLimit); dof.body.xDrive = d; } else if (dof.axis == 1) { var d = dof.body.yDrive; d.target = Mathf.Clamp(d.target + delta, d.lowerLimit, d.upperLimit); dof.body.yDrive = d; } else { var d = dof.body.zDrive; d.target = Mathf.Clamp(d.target + delta, d.lowerLimit, d.upperLimit); dof.body.zDrive = d; } }
    }

    [Serializable] class SkinTrace { public SkinRow[] rows; }
    [Serializable] class EpisodeTrace { public EpisodeRow[] rows; }
    [Serializable] class JacobianTrace { public JacobianRow[] rows; }
    [Serializable] class JacobianRow { public int step, rows, columns; public int[] selected_columns; public float damping, error_m, normal_determinant; public Vector3 wrist_m, waypoint_m; public bool collision_stop; }
    [Serializable] class ReachAudit { public string schema; public Vector3 target_position_m; public bool target_disabled_calibration_geometry; public ReachAuditRow[] rows; }
    [Serializable] class ReachAuditRow { public float lower_target_deg, target_center_distance_m; public Vector3 shoulder_xyz_target_deg, wrist_m, fingertip_centroid_m; }
    [Serializable] class ArmCalibration { public string schema, scene_spec_path; public bool target_disabled_calibration_geometry; public Vector3 waypoint_m, final_wrist_m; public string[] bodies; public int[] axes; public float[] targets_deg; public float final_error_m; public CalibrationRow[] history; }
    [Serializable] class CalibrationRow { public string body; public int axis; public float increment_deg, baseline_error_m, plus_error_m, minus_error_m, chosen_target_deg, chosen_error_m; }
    [Serializable] class FreshCandidate { public string schema, scene_spec_path, initial_state_hash, final_state_hash; public int fixed_steps, fixed_hz; public bool target_disabled_calibration_geometry; public float[] targets_deg; public Vector3 waypoint_m, final_wrist_m; public float final_error_m; }
    [Serializable] class FreshCalibrationAggregate { public float[] frozen_targets_deg; }
    [Serializable] class SceneInput { public SceneTarget target; public SceneInstance[] instances; }
    [Serializable] class SceneTarget { public SceneTransform transform; }
    [Serializable] class SceneInstance { public string persistent_id; public SceneTransform transform; }
    [Serializable] class SceneTransform { public float[] position_m, scale_m; public float rotation_y_deg; }
    [Serializable] class EpisodeRow { public int step, digit_contacts; public float time_s, contact_impulse_n_s, lift_m, finger_penetration_m, support_penetration_m; public string phase, prior_contact_digits, contact_digits; public Vector3 object_position, object_velocity; public bool collision_stop_command, qualified; }
    [Serializable] class EpisodeReport { public string schema, scene_spec_path; public int max_digit_contacts, assistance_entries; public float max_lift_m, stabilization_drift_m, maximum_support_penetration_m, initial_finger_penetration_m; public Vector3 settled_object_position, final_object_position; public bool contact_during_stabilization, valid_initialization, qualified, free_dynamic_target, attachment_or_pose_writes, passed; }
    [Serializable] class StabilizationReceipt { public string schema, unity_version, physics_backend, scene_spec_path; public int fixed_hz, observed_steps, post_settle_window_steps; public float post_settle_drift_m, maximum_support_penetration_m; public Vector3 target_start_position, target_settled_position; public bool passed; }
    [Serializable] class DrivenApproachReceipt { public string schema, scene_spec_path; public Vector3 initial_shoulder_position_m, initial_wrist_position_m, initial_fingertip_centroid_m, driven_shoulder_position_m, driven_wrist_position_m, target_position_m; public float minimum_fingertip_surface_distance_m; public bool overlap_before_closure, actuator_drives_only, target_pose_writes_after_initialization, passed; }
    [Serializable] class SkinRow { public int step; public Bounds mesh_bounds; public float segment_length_max_error_m; public float[] collider_skin_errors_m, head_joint_position, head_drive_targets_deg; public Vector3 camera_position, camera_forward; public bool finite; }
    [Serializable] class FovMetric { public int fov_deg, visible_fingertips; public bool target_visible, wrist_visible; public Vector3 target_center_viewport, wrist_viewport; public Vector3[] fingertip_viewports; }
    [Serializable] class SkinReport { public string schema, unity_version, head_axis_audit, camera_skin_clearance_method; public string[] selected_articulation_bodies, excluded_adjacent_collision_pairs, retained_distal_collision_pairs; public float[] head_joint_position, head_drive_targets_deg, neck_joint_position, torso_joint_position; public Vector3[] fingertip_positions; public FovMetric[] fov_metrics; public bool actual_weighted_skin, articulation_is_only_pose_state, independently_advanced_animation, all_finite, qa_colliders_in_clean_or_pov, passed; public int fixed_physics_hz, frozen_fov_deg; public float segment_length_max_error_m, mesh_bounds_ratio_max, collider_skin_median_m, collider_skin_p95_m, collider_skin_max_m, camera_clearance_m, neutral_face_forward_angle_deg, neck_drive_target_deg, camera_roll_deg; public Vector3 camera_mount_local_position, camera_mount_local_euler, target_position, wrist_position, final_camera_forward, final_camera_up, camera_to_target_direction; }
}

[ExecuteAlways]
public sealed class EpisodeContactProbe : MonoBehaviour
{
    public readonly HashSet<int> digits = new(); public float impulse;
    public void BeginStep() { digits.Clear(); impulse = 0; }
    void OnCollisionStay(Collision collision) { var name = collision.gameObject.name; if (!name.StartsWith("finger")) return; if (int.TryParse(name.Substring(6, 1), out int digit)) digits.Add(digit); impulse += collision.impulse.magnitude; }
    void OnCollisionEnter(Collision collision) { OnCollisionStay(collision); }
}
