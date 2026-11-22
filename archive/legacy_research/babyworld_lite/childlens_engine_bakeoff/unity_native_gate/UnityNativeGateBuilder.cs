using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class UnityNativeGateBuilder
{
    const int PhysicsHz = 240;
    const float Dt = 1f / PhysicsHz;
    static readonly string Output = Environment.GetEnvironmentVariable("UNITY_NATIVE_GATE_OUTPUT");
    static readonly List<ArticulationBody> joints = new();
    static readonly List<ContactProbe> probes = new();
    static ContactProbe targetProbe;

    [MenuItem("BabyWorld/Run Unity Native Preflight")]
    public static void RunPreflight()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_NATIVE_GATE_OUTPUT is required");
        Directory.CreateDirectory(Output);
        BuildCell();
        Physics.simulationMode = SimulationMode.Script;
        Physics.defaultSolverIterations = 24;
        Physics.defaultSolverVelocityIterations = 12;
        var target = GameObject.Find("free_target").GetComponent<Rigidbody>();
        var targetCollider = target.GetComponent<Collider>();
        bool initialOverlap = DigitPenetrations(targetCollider).Any(x => x > 0);
        RenderDebug("open_initial.png");
        var rows = new List<TraceRow>();
        for (int step = 0; step < 720; step++) {
            float close = Mathf.SmoothStep(0, 1, Mathf.InverseLerp(240, 480, step));
            foreach (var joint in joints) {
                var drive = joint.xDrive;
                int digit = int.Parse(joint.name.Split('_')[1]);
                drive.target = (digit <= 2 ? 42f : -42f) * close;
                drive.stiffness = 18f;
                drive.damping = 3.5f;
                drive.forceLimit = 1.2f;
                joint.xDrive = drive;
            }
            Physics.Simulate(Dt);
            Physics.SyncTransforms();
            var penetration = DigitPenetrations(targetCollider);
            rows.Add(new TraceRow {
                step = step, time_s = step * Dt, target_position = target.position,
                target_velocity = target.linearVelocity, target_angular_velocity = target.angularVelocity,
                first_joint_euler = joints[0].transform.eulerAngles,
                digit_contacts = targetProbe.contactDigits.Count,
                contact_impulse_n_s = targetProbe.lastImpulse,
                geometric_digit_contacts = penetration.Count(x => x > 0),
                max_penetration_m = penetration.Max(),
                digit_surface_distance = Enumerable.Range(1, 5).Select(d => probes.Where(p => p.digit == d).Min(p => Vector3.Distance(p.GetComponent<Collider>().ClosestPoint(target.position), target.position))).ToArray(),
                joint_position = joints.SelectMany(j => Values(j.jointPosition)).ToArray(),
                joint_velocity = joints.SelectMany(j => Values(j.jointVelocity)).ToArray(),
                joint_acceleration = joints.SelectMany(j => Values(j.jointAcceleration)).ToArray(),
                joint_force = joints.SelectMany(j => Values(j.jointForce)).ToArray(),
                drive_force = joints.SelectMany(j => Values(j.driveForce)).ToArray()
            });
            foreach (var p in probes) p.BeginStep(); targetProbe.BeginStep();
            if (step == 479) RenderDebug("closed_contact.png");
        }
        var maxContacts = rows.Max(r => r.digit_contacts);
        var sustained = rows.Skip(480).Count(r => r.digit_contacts >= 3 && r.geometric_digit_contacts >= 3);
        float stabilizationDrift = Vector3.Distance(rows[0].target_position, rows[239].target_position);
        var report = new Report {
            schema = "embodied.unity_native.preflight.v1", unity_version = Application.unityVersion,
            graphics_api = SystemInfo.graphicsDeviceType.ToString(), architecture = SystemInfo.processorType,
            physics_authority = "Unity ArticulationBody/PhysX only", physics_hz = PhysicsHz,
            fixed_manual_stepping = true, target_free_dynamic_rigidbody = true,
            articulation_count = joints.Count + 1, revolute_joint_count = joints.Count,
            max_simultaneous_digit_contacts = maxContacts, sustained_three_digit_steps = sustained,
            initial_digit_overlap = initialOverlap, stabilization_drift_m = stabilizationDrift,
            callback_geometric_contact_agreement_steps = rows.Count(r => r.digit_contacts == r.geometric_digit_contacts),
            maximum_digit_penetration_m = rows.Max(r => r.max_penetration_m),
            collision_impulse_api_observed = rows.Any(r => r.contact_impulse_n_s > 0),
            joint_position_recorded = rows.Any(r => r.joint_position.Length > 0),
            joint_velocity_recorded = rows.Any(r => r.joint_velocity.Length > 0),
            joint_acceleration_recorded = rows.Any(r => r.joint_acceleration.Length > 0),
            joint_force_recorded = rows.Any(r => r.joint_force.Length > 0),
            drive_force_recorded = rows.Any(r => r.drive_force.Length > 0),
            observed_collision_pairs = ContactProbe.collisionPairs.OrderBy(x => x).ToArray(),
            passed = !initialOverlap && stabilizationDrift <= .001f && maxContacts >= 3 && sustained >= 24 &&
                     rows.Skip(240).Any(r => r.contact_impulse_n_s > 0) && rows.Max(r => r.max_penetration_m) <= .003f
        };
        File.WriteAllText(Path.Combine(Output, "preflight_trace.json"), JsonUtility.ToJson(new Trace { rows = rows.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "preflight_report.json"), JsonUtility.ToJson(report, true));
        EditorApplication.Exit(report.passed ? 0 : 2);
    }

    static void BuildCell()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
            UnityEngine.Object.DestroyImmediate(o);
        joints.Clear(); probes.Clear();
        var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
        floor.name = "support"; floor.transform.position = new Vector3(0, -.025f, 0); floor.transform.localScale = new Vector3(1, .05f, 1);
        var root = new GameObject("articulation_palm_root");
        root.transform.position = new Vector3(0, .10f, 0);
        var rootBody = root.AddComponent<ArticulationBody>(); rootBody.immovable = true;
        var palmCollider = root.AddComponent<BoxCollider>(); palmCollider.size = new Vector3(.12f, .035f, .10f);
        var target = GameObject.CreatePrimitive(PrimitiveType.Cube); target.name = "free_target";
        target.transform.position = new Vector3(0, .145f, 0); target.transform.localScale = Vector3.one * .055f;
        var rb = target.AddComponent<Rigidbody>(); rb.mass = .055f; rb.useGravity = true; rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        rb.maxDepenetrationVelocity = 1.0f; target.GetComponent<Collider>().material = NewMaterial(.9f, .05f);
        targetProbe = target.AddComponent<ContactProbe>(); targetProbe.isTarget = true;
        float[] x = { -.022f, -.011f, 0f, .011f, .022f };
        for (int digit = 0; digit < 5; digit++) {
            Transform parent = root.transform;
            for (int part = 0; part < 3; part++) {
                var segment = new GameObject($"digit_{digit + 1}_phalanx_{part + 1}");
                segment.transform.SetParent(parent, false);
                segment.transform.localPosition = part == 0
                    ? new Vector3(x[digit], .045f, digit < 2 ? -.065f : .065f)
                    : new Vector3(0, .028f, 0);
                // The thumb opposes the four fingers.  All joints retain the
                // same bounded positive flexion command in their local frame.
                segment.transform.localRotation = Quaternion.identity;
                var body = segment.AddComponent<ArticulationBody>(); body.jointType = ArticulationJointType.RevoluteJoint;
                body.anchorRotation = Quaternion.identity;
                body.twistLock = ArticulationDofLock.LimitedMotion;
                var drive = body.xDrive; drive.lowerLimit = -55; drive.upperLimit = 55; drive.stiffness = 18; drive.damping = 3.5f; drive.forceLimit = 1.2f; body.xDrive = drive;
                var collider = segment.AddComponent<CapsuleCollider>(); collider.direction = 1; collider.radius = part == 0 ? .006f : .005f; collider.height = part == 0 ? .070f : .032f;
                collider.center = new Vector3(0, part == 0 ? .035f : .016f, 0); collider.material = NewMaterial(.9f, .05f);
                var visual = GameObject.CreatePrimitive(PrimitiveType.Capsule); visual.name = "QA_COLLIDER_VISUAL_ONLY";
                UnityEngine.Object.DestroyImmediate(visual.GetComponent<Collider>()); visual.transform.SetParent(segment.transform, false);
                visual.transform.localPosition = collider.center; visual.transform.localScale = new Vector3(collider.radius * 2, collider.height * .5f, collider.radius * 2);
                var qaMaterial = new Material(Shader.Find("Standard")); qaMaterial.color = digit < 2 ? new Color(.1f, .75f, 1f) : new Color(1f, .45f, .1f); visual.GetComponent<Renderer>().sharedMaterial = qaMaterial;
                var probe = segment.AddComponent<ContactProbe>(); probe.digit = digit + 1; probes.Add(probe);
                joints.Add(body); parent = segment.transform;
            }
        }
        foreach (var a in probes) foreach (var b in probes) if (a != b) {
            bool sameDigit = a.digit == b.digit;
            bool bothDistal = a.name.EndsWith("_3") && b.name.EndsWith("_3");
            if (sameDigit || !bothDistal) Physics.IgnoreCollision(a.GetComponent<Collider>(), b.GetComponent<Collider>(), true);
        }
        Physics.IgnoreCollision(floor.GetComponent<Collider>(), palmCollider, true);
    }

    static PhysicsMaterial NewMaterial(float friction, float bounce) => new PhysicsMaterial { dynamicFriction = friction, staticFriction = friction, bounciness = bounce, frictionCombine = PhysicsMaterialCombine.Maximum };
    static IEnumerable<float> Values(ArticulationReducedSpace value) { for (int i = 0; i < value.dofCount; i++) yield return value[i]; }
    static float[] DigitPenetrations(Collider target) => Enumerable.Range(1, 5).Select(d => probes.Where(p => p.digit == d).Select(p => {
        var c = p.GetComponent<Collider>(); return Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, target, target.transform.position, target.transform.rotation, out _, out float distance) ? distance : 0f;
    }).Max()).ToArray();
    static void RenderDebug(string name) {
        var cameraObject = new GameObject("stage_a_debug_camera");
        var camera = cameraObject.AddComponent<Camera>(); camera.transform.position = new Vector3(.34f, .28f, .34f); camera.transform.LookAt(new Vector3(0, .15f, 0)); camera.fieldOfView = 34; camera.nearClipPlane = .02f;
        var light = new GameObject("stage_a_debug_light").AddComponent<Light>(); light.type = LightType.Directional; light.intensity = 1.2f; light.transform.rotation = Quaternion.Euler(45, -30, 0);
        var rt = new RenderTexture(640, 360, 24); var tex = new Texture2D(640, 360, TextureFormat.RGB24, false); camera.targetTexture = rt; camera.Render(); RenderTexture.active = rt; tex.ReadPixels(new Rect(0, 0, 640, 360), 0, 0); tex.Apply(); File.WriteAllBytes(Path.Combine(Output, name), tex.EncodeToPNG()); RenderTexture.active = null;
        UnityEngine.Object.DestroyImmediate(rt); UnityEngine.Object.DestroyImmediate(tex); UnityEngine.Object.DestroyImmediate(cameraObject); UnityEngine.Object.DestroyImmediate(light.gameObject);
    }

    [Serializable] class Trace { public TraceRow[] rows; }
    [Serializable] class TraceRow { public int step; public float time_s; public Vector3 target_position, target_velocity, target_angular_velocity, first_joint_euler; public int digit_contacts, geometric_digit_contacts; public float contact_impulse_n_s, max_penetration_m; public float[] digit_surface_distance, joint_position, joint_velocity, joint_acceleration, joint_force, drive_force; }
    [Serializable] class Report { public string schema, unity_version, graphics_api, architecture, physics_authority; public string[] observed_collision_pairs; public int physics_hz, articulation_count, revolute_joint_count, max_simultaneous_digit_contacts, sustained_three_digit_steps, callback_geometric_contact_agreement_steps; public float stabilization_drift_m, maximum_digit_penetration_m; public bool initial_digit_overlap, fixed_manual_stepping, target_free_dynamic_rigidbody, collision_impulse_api_observed, joint_position_recorded, joint_velocity_recorded, joint_acceleration_recorded, joint_force_recorded, drive_force_recorded, passed; }
}

[ExecuteAlways]
public sealed class ContactProbe : MonoBehaviour
{
    public static readonly HashSet<string> collisionPairs = new();
    public int digit; public bool isTarget, targetContact; public float lastImpulse; public readonly HashSet<int> contactDigits = new();
    public void BeginStep() { targetContact = false; lastImpulse = 0; contactDigits.Clear(); }
    void OnCollisionStay(Collision collision) {
        collisionPairs.Add(name + "->" + collision.gameObject.name);
        var other = collision.gameObject.GetComponent<ContactProbe>();
        if (collision.gameObject.name == "free_target" || (isTarget && other && other.digit > 0)) {
            targetContact = true; lastImpulse += collision.impulse.magnitude;
            if (isTarget && other) contactDigits.Add(other.digit);
        }
    }
    void OnCollisionEnter(Collision collision) { OnCollisionStay(collision); }
}
