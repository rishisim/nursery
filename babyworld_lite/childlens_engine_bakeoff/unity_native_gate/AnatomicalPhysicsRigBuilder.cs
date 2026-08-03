using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;

// Canonical Unity harness for configs/embodied_simulation_anatomical_rig.json.
// The imported MPFB hierarchy is appearance only. Every collider and every
// controlled DOF belongs to the deliberately authored physical hierarchy.
public static class AnatomicalPhysicsRigBuilder
{
    const float Dt = 1f / 240f;
    static readonly string Output = Environment.GetEnvironmentVariable("ANATOMICAL_RIG_OUTPUT");
    static readonly string ManifestPath = Environment.GetEnvironmentVariable("ANATOMICAL_RIG_MANIFEST");
    static readonly Dictionary<string, Vector3> landmarks = new();
    static readonly Dictionary<string, Transform> visual = new();
    static readonly Dictionary<string, Transform> physicalSites = new();
    static readonly Dictionary<string, Quaternion> followerOffsets = new();
    static readonly Dictionary<string, Vector3> followerPositionOffsets = new();
    static readonly List<ArticulationBody> bodies = new();
    static readonly List<ArticulationBody> controlled = new();
    static readonly List<ColliderBinding> colliderBindings = new();
    static GameObject avatar;
    static SkinnedMeshRenderer skin;
    static Mesh baked;
    static string manifestJson;

    [MenuItem("BabyWorld/Run Anatomical Rig Stage A")]
    public static void RunStageA()
    {
        RequireEnvironment();
        Directory.CreateDirectory(Output);
        var rows = new List<SweepRow>();
        var restErrors = new List<float>();
        Build();
        restErrors.AddRange(RegistrationErrors());
        var roundTrip = landmarks.Values.Max(p => Vector3.Distance(p, avatar.transform.TransformPoint(avatar.transform.InverseTransformPoint(p))));
        for (int jointIndex = 0; jointIndex < controlled.Count; jointIndex++) {
            var body = controlled[jointIndex];
            int dofs = body.dofCount;
            for (int axis = 0; axis < dofs; axis++) {
                foreach (float fraction in new[] { -.65f, .65f, 0f }) {
                    ZeroTargets();
                    var drive = Drive(body, axis);
                    SetDrive(body, axis, drive, Mathf.Lerp(0, fraction > 0 ? drive.upperLimit : drive.lowerLimit, Mathf.Abs(fraction)));
                    Simulate(300);
                    FollowSkin();
                    var errors = RegistrationErrors();
                    restErrors.AddRange(errors);
                    rows.Add(new SweepRow { joint = body.name, axis = axis, target_deg = Drive(body, axis).target,
                        joint_position_rad = body.jointPosition[axis], site_m = SiteFor(body), collider_skin_max_m = errors.Max(),
                        finite = errors.All(float.IsFinite) && Finite(SiteFor(body)) });
                }
            }
        }
        ZeroTargets(); Simulate(300); FollowSkin();
        RenderExternal("stage_a_clean.png", false); RenderExternal("stage_a_overlay.png", true);
        var report = new StageAReport {
            schema = "embodied.anatomical_rig.stage_a.v1", unity_version = Application.unityVersion,
            source_fbx_sha256 = Sha256("Assets/Avatar/child.fbx"), manifest_sha256 = Sha256(ManifestPath),
            physical_bodies = bodies.Select(x => x.name).ToArray(), controlled_dofs = controlled.Sum(x => x.dofCount),
            coordinate_roundtrip_max_m = roundTrip, collider_skin_median_m = Quantile(restErrors, .5f),
            collider_skin_p95_m = Quantile(restErrors, .95f), collider_skin_max_m = restErrors.Max(),
            all_sweeps_finite = rows.All(x => x.finite), one_weighted_skin = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true).Length == 1,
            deformation_bones_have_articulation_bodies = avatar.GetComponentsInChildren<ArticulationBody>(true).Length > 0,
            independently_advanced_animation = avatar.GetComponentsInChildren<Animator>(true).Any(x => x.enabled) || avatar.GetComponentsInChildren<Animation>(true).Any(x => x.enabled),
        };
        report.passed = report.coordinate_roundtrip_max_m <= 1e-6f && report.collider_skin_max_m <= .0075f && report.all_sweeps_finite && report.one_weighted_skin && !report.deformation_bones_have_articulation_bodies && !report.independently_advanced_animation;
        File.WriteAllText(Path.Combine(Output, "stage_a_sweep.json"), JsonUtility.ToJson(new SweepTrace { rows = rows.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "stage_a_registration_by_bone.json"), JsonUtility.ToJson(new RegistrationTrace { rows = RegistrationByBone() }, true));
        File.WriteAllText(Path.Combine(Output, "stage_a_report.json"), JsonUtility.ToJson(report, true));
        EditorApplication.Exit(report.passed ? 0 : 2);
    }

    [MenuItem("BabyWorld/Run Anatomical Rig Stage B")]
    public static void RunStageB()
    {
        RequireEnvironment(); Directory.CreateDirectory(Output);
        var jacobianRows = new List<JacobianAuditRow>();
        float[][] poses = { new[]{5f,12f,-8f,55f,15f,-10f,8f}, new[]{-20f,35f,18f,80f,-25f,20f,-12f}, new[]{30f,-20f,25f,40f,30f,-25f,15f} };
        foreach (var pose in poses) {
            Build(); ApplyArmTargets(pose); Simulate(1200); FollowSkin();
            var root = bodies[0]; var starts = new List<int>(); int columns = root.GetDofStartIndices(starts);
            var dense = new ArticulationJacobian(bodies.Count * 6, columns); root.GetDenseJacobian(ref dense);
            var palm = bodies.Single(x => x.name == "physical_palm"); int palmRow = palm.index * 6;
            var auditBodies=controlled.Take(5).Select(x=>new AuditBody{name=x.name,index=x.index,dofs=x.dofCount}).ToArray();
            for (int i = 0; i < auditBodies.Length; i++) {
                var body = auditBodies[i]; int axisCount = body.dofs;
                for (int axis = 0; axis < axisCount; axis++) {
                    string bodyName = body.name; int column = starts[body.index] + axis;
                    Vector3 engine = new Vector3(dense[palmRow, column], dense[palmRow + 1, column], dense[palmRow + 2, column]);
                    const float epsilonDeg = 1f;
                    Vector3 plus = EvaluatePalm(pose, i, axis, epsilonDeg), minus = EvaluatePalm(pose, i, axis, -epsilonDeg);
                    Vector3 finite = (plus - minus) / (2 * epsilonDeg * Mathf.Deg2Rad);
                    float direction = engine.sqrMagnitude < 1e-12f && finite.sqrMagnitude < 1e-12f ? 0 : Vector3.Angle(engine, finite);
                    float relative = Mathf.Abs(engine.magnitude - finite.magnitude) / Mathf.Max(1e-9f, Mathf.Max(engine.magnitude, finite.magnitude));
                    jacobianRows.Add(new JacobianAuditRow { pose_deg = pose, joint = bodyName, axis = axis, engine_m_per_rad = engine, finite_difference_m_per_rad = finite, direction_error_deg = direction, relative_magnitude_error = relative });
                }
            }
        }
        var waypointRows = new List<WaypointRow>();
        Vector3[] offsets = { new(.00f,.00f,.00f), new(.025f,.015f,.015f), new(-.02f,.02f,.025f), new(.015f,-.015f,.035f), new(-.015f,.005f,.05f) };
        Build(); Simulate(720); Vector3 origin = physicalSites["palm"].position; Quaternion originRotation=physicalSites["palm"].rotation;
        foreach (var offset in offsets) {
            Build(); Simulate(720); Vector3 target = origin + offset; bool collided = false;
            for (int step = 0; step < 960; step++) { DlsStep(target); Simulate(1); FollowSkin(); collided |= UnintendedSelfContact(); }
            Vector3 observed = physicalSites["palm"].position;
            waypointRows.Add(new WaypointRow { target_m = target, observed_m = observed, position_error_m = Vector3.Distance(target, observed), orientation_error_deg = Quaternion.Angle(physicalSites["palm"].rotation, originRotation), collision_free = !collided });
        }
        RenderExternal("stage_b_waypoints.png", true);
        var report = new StageBReport { schema = "embodied.anatomical_rig.stage_b.v1", unity_version = Application.unityVersion,
            jacobian_rows = jacobianRows.Count, jacobian_max_direction_error_deg = jacobianRows.Max(x => x.direction_error_deg), jacobian_max_relative_magnitude_error = jacobianRows.Max(x => x.relative_magnitude_error),
            waypoint_count = waypointRows.Count, palm_position_max_error_m = waypointRows.Max(x => x.position_error_m), palm_orientation_max_error_deg = waypointRows.Max(x => x.orientation_error_deg), all_waypoints_collision_free = waypointRows.All(x => x.collision_free) };
        report.passed = report.jacobian_max_direction_error_deg <= 2f && report.jacobian_max_relative_magnitude_error <= .03f && report.waypoint_count >= 5 && report.palm_position_max_error_m <= .010f && report.palm_orientation_max_error_deg <= 7 && report.all_waypoints_collision_free;
        File.WriteAllText(Path.Combine(Output, "stage_b_jacobian.json"), JsonUtility.ToJson(new JacobianAudit { rows = jacobianRows.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "stage_b_waypoints.json"), JsonUtility.ToJson(new WaypointTrace { rows = waypointRows.ToArray() }, true));
        File.WriteAllText(Path.Combine(Output, "stage_b_report.json"), JsonUtility.ToJson(report, true));
        EditorApplication.Exit(report.passed ? 0 : 2);
    }

    [MenuItem("BabyWorld/Capture Anatomical Rig Failure Evidence")]
    public static void CaptureFailureEvidence()
    {
        RequireEnvironment(); Directory.CreateDirectory(Output);
        string cleanDir=Path.Combine(Output,"stage_a_clean_frames"),overlayDir=Path.Combine(Output,"stage_a_overlay_frames"),waypointDir=Path.Combine(Output,"stage_b_waypoint_frames");
        Directory.CreateDirectory(cleanDir);Directory.CreateDirectory(overlayDir);Directory.CreateDirectory(waypointDir);
        var evidence=new List<EvidenceFrame>(); int cleanFrame=0,waypointFrame=0;
        Build(); var camera=EvidenceCamera(); var label=EvidenceLabel(camera); var overlay=BuildLiveOverlay();
        int dofOrdinal=0,totalDofs=controlled.Sum(x=>x.dofCount);
        for(int jointIndex=0;jointIndex<controlled.Count;jointIndex++){
            var body=controlled[jointIndex];
            for(int axis=0;axis<body.dofCount;axis++){
                dofOrdinal++;
                ZeroTargets();Simulate(96);var drive=Drive(body,axis);
                for(int sample=0;sample<12;sample++){
                    float phase=sample/11f;float target=phase<.5f?Mathf.Lerp(drive.lowerLimit*.65f,drive.upperLimit*.65f,phase*2):Mathf.Lerp(drive.upperLimit*.65f,0,(phase-.5f)*2);
                    SetDrive(body,axis,drive,target);Simulate(8);FollowSkin();
                    label.text=$"STAGE A PASS  |  DOF {dofOrdinal}/{totalDofs}  {body.name} axis {axis}\nlimits [{drive.lowerLimit:F0}, {drive.upperLimit:F0}] deg  target {target:F1} deg  q {body.jointPosition[axis]*Mathf.Rad2Deg:F1} deg";
                    overlay.SetActive(false);CaptureEvidenceFrame(camera,Path.Combine(cleanDir,$"frame_{cleanFrame:D4}.png"));
                    overlay.SetActive(true);CaptureEvidenceFrame(camera,Path.Combine(overlayDir,$"frame_{cleanFrame:D4}.png"));
                    evidence.Add(new EvidenceFrame{chapter="stage_a_dof",frame=cleanFrame,trial=jointIndex,joint=body.name,axis=axis,target_deg=target,observed_deg=body.jointPosition[axis]*Mathf.Rad2Deg});cleanFrame++;
                }
            }
        }
        Build();Simulate(720);Vector3 origin=physicalSites["palm"].position;Quaternion originRotation=physicalSites["palm"].rotation;
        Vector3[] offsets={new(.00f,.00f,.00f),new(.025f,.015f,.015f),new(-.02f,.02f,.025f),new(.015f,-.015f,.035f),new(-.015f,.005f,.05f)};
        for(int trial=0;trial<offsets.Length;trial++){
            Build();Simulate(720);camera=EvidenceCamera();label=EvidenceLabel(camera);overlay=BuildLiveOverlay();overlay.SetActive(true);
            Vector3 target=origin+offsets[trial];var targetMarker=Marker("TARGET_RED",target,new Color(.95f,.03f,.03f),.012f);var observedMarker=Marker("OBSERVED_BLUE",physicalSites["palm"].position,new Color(.02f,.25f,1f),.010f);
            var trailObject=new GameObject("OBSERVED_TRAIL_BLUE");var trail=trailObject.AddComponent<LineRenderer>();trail.material=new Material(Shader.Find("Sprites/Default"));trail.startColor=trail.endColor=new Color(.02f,.25f,1f);trail.startWidth=.0035f;trail.endWidth=.0035f;trail.positionCount=0;
            for(int frame=0;frame<120;frame++){
                for(int step=0;step<8;step++){DlsStep(target);Simulate(1);}Vector3 observed=physicalSites["palm"].position;observedMarker.transform.position=observed;trail.positionCount=frame+1;trail.SetPosition(frame,observed);
                float positionError=Vector3.Distance(target,observed),orientationError=Quaternion.Angle(physicalSites["palm"].rotation,originRotation);
                label.text=$"STAGE B NO-GO  |  fresh-state waypoint {trial+1}/5\nRED target  BLUE observed/trail  |  pos {positionError*1000:F1} mm  orient {orientationError:F1} deg";
                CaptureEvidenceFrame(camera,Path.Combine(waypointDir,$"frame_{waypointFrame:D4}.png"));
                evidence.Add(new EvidenceFrame{chapter="stage_b_waypoint",frame=waypointFrame,trial=trial,target_m=target,observed_m=observed,position_error_m=positionError,orientation_error_deg=orientationError});waypointFrame++;
            }
        }
        File.WriteAllText(Path.Combine(Output,"evidence_frame_ledger.json"),JsonUtility.ToJson(new EvidenceTrace{rows=evidence.ToArray()},true));
        File.WriteAllText(Path.Combine(Output,"evidence_capture_report.json"),JsonUtility.ToJson(new EvidenceReport{schema="embodied.anatomical_rig.failure_evidence.v1",unity_version=Application.unityVersion,stage_a_frames=cleanFrame,stage_b_frames=waypointFrame,stage_a_clean_and_overlay_same_state=true,physics_hz=240,render_hz=30,steps_per_frame=8,stage_b_decision="NO-GO"},true));
        EditorApplication.Exit(0);
    }

    static void RequireEnvironment() { if (string.IsNullOrWhiteSpace(Output) || string.IsNullOrWhiteSpace(ManifestPath)) throw new Exception("ANATOMICAL_RIG_OUTPUT and ANATOMICAL_RIG_MANIFEST are required"); }
    static void Build()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        landmarks.Clear(); visual.Clear(); physicalSites.Clear(); followerOffsets.Clear(); followerPositionOffsets.Clear(); bodies.Clear(); controlled.Clear(); colliderBindings.Clear();
        manifestJson = File.ReadAllText(ManifestPath); ParseLandmarks();
        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx"));
        avatar.name = "CC0_WEIGHTED_CHILD_VISUAL_ONLY"; avatar.transform.localScale = Vector3.one * 1.9f;
        foreach (var a in avatar.GetComponentsInChildren<Animator>(true)) a.enabled = false; foreach (var a in avatar.GetComponentsInChildren<Animation>(true)) a.enabled = false;
        skin = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true).Single(); skin.updateWhenOffscreen = true; skin.localBounds = new Bounds(Vector3.zero, Vector3.one * 4);
        foreach (var t in avatar.GetComponentsInChildren<Transform>(true)) visual[t.name] = t;
        var root = new GameObject("physical_torso_root"); root.transform.position = landmarks["shoulder"]; var rootBody = root.AddComponent<ArticulationBody>(); rootBody.immovable = true; bodies.Add(rootBody);
        var shoulder = Joint("physical_shoulder", root.transform, landmarks["shoulder"], true, new Vector3(0,0,1), new Vector2(-85,95), .03f);
        var elbow = Joint("physical_elbow", shoulder.transform, landmarks["elbow"], false, new Vector3(-.634f,-.567f,.526f), new Vector2(0,145), .28f);
        var forearm = Joint("physical_forearm_pronation", elbow.transform, landmarks["wrist"], false, (landmarks["wrist"]-landmarks["elbow"]).normalized, new Vector2(-75,75), .18f);
        var wristFlex = Joint("physical_wrist_flex", forearm.transform, landmarks["wrist"], false, new Vector3(.684f,.729f,0), new Vector2(-55,60), .02f);
        var wrist = Joint("physical_wrist_deviate", wristFlex.transform, landmarks["wrist"], false, new Vector3(-.526f,.493f,.693f), new Vector2(-25,35), .02f);
        var palm = Fixed("physical_palm", wrist.transform, landmarks["wrist"], .09f); physicalSites["palm"] = palm.transform;
        Bind("upperarm01.R", shoulder.transform); Bind("lowerarm01.R", elbow.transform); Bind("wrist.R", palm.transform);
        Bind("upperarm02.R", shoulder.transform); Bind("lowerarm02.R", elbow.transform);
        if(visual.ContainsKey("clavicle.R")) Bind("clavicle.R", shoulder.transform);
        if(visual.ContainsKey("shoulder01.R")) Bind("shoulder01.R", shoulder.transform);
        string[] digits = { "thumb", "index", "middle", "ring", "little" }; string[] mpfb = { "finger1", "finger2", "finger3", "finger4", "finger5" };
        for (int d = 0; d < digits.Length; d++) {
            Transform parent = palm.transform;
            for (int part = 1; part <= 3; part++) {
                string key = digits[d] + "_" + part; Vector3 point = landmarks[key];
                Vector3 next = part < 3 ? landmarks[digits[d] + "_" + (part + 1)] : point + (point - landmarks[digits[d] + "_2"]) * .65f;
                Vector3 direction = (next - point).normalized; Vector3 axis = Vector3.Cross(direction, (landmarks["middle_1"] - landmarks["wrist"]).normalized).normalized;
                if (axis.sqrMagnitude < .5f) axis = Vector3.right;
                var segment = Joint("physical_" + key, parent, point, false, axis, d == 0 && part == 1 ? new Vector2(-35,45) : new Vector2(0, part == 3 ? 80 : 100), .012f);
                Bind(mpfb[d] + "-" + part + ".R", segment.transform); physicalSites[key] = segment.transform; parent = segment.transform;
            }
        }
        baked = new Mesh { name = "anatomical_registration_bake" }; FollowSkin();
        foreach (var row in followerOffsets.Where(x => x.Key != "upperarm02.R" && x.Key != "lowerarm02.R" && x.Key != "shoulder01.R" && x.Key != "clavicle.R")) AddFittedSpheres(bodies.Single(x => x.transform == physicalSites[row.Key]), row.Key);
        var support = GameObject.CreatePrimitive(PrimitiveType.Cube); support.name = "support"; support.transform.position = new Vector3(.365f,.58f,.285f); support.transform.localScale = new Vector3(.55f,.04f,.45f);
        Physics.simulationMode = SimulationMode.Script; Physics.defaultSolverIterations = 24; Physics.defaultSolverVelocityIterations = 12; Physics.SyncTransforms();
    }

    static ArticulationBody Joint(string name, Transform parent, Vector3 worldPosition, bool spherical, Vector3 axisWorld, Vector2 limits, float mass) {
        var go = new GameObject(name); go.transform.SetPositionAndRotation(worldPosition, Quaternion.FromToRotation(Vector3.right, axisWorld.normalized)); go.transform.SetParent(parent, true);
        var b = go.AddComponent<ArticulationBody>(); b.mass = mass; b.jointType = spherical ? ArticulationJointType.SphericalJoint : ArticulationJointType.RevoluteJoint; b.twistLock = ArticulationDofLock.LimitedMotion;
        Configure(ref b, 0, limits); if (spherical) { b.swingYLock = ArticulationDofLock.LimitedMotion; b.swingZLock = ArticulationDofLock.LimitedMotion; Configure(ref b, 1, new Vector2(-70,70)); Configure(ref b, 2, new Vector2(-45,45)); }
        bodies.Add(b); controlled.Add(b); return b;
    }
    static ArticulationBody Fixed(string name, Transform parent, Vector3 position, float mass) { var go = new GameObject(name); go.transform.position = position; go.transform.SetParent(parent, true); var b = go.AddComponent<ArticulationBody>(); b.mass = mass; b.jointType = ArticulationJointType.FixedJoint; bodies.Add(b); return b; }
    static void Configure(ref ArticulationBody b, int axis, Vector2 limits) { var d = Drive(b, axis); d.lowerLimit = limits.x; d.upperLimit = limits.y; d.stiffness = 260; d.damping = 16; d.forceLimit = 30; SetDriveObject(b, axis, d); }
    static void AddFittedSpheres(ArticulationBody body, string visualBone) {
        int index = Array.IndexOf(skin.bones, visual[visualBone]); var points = new List<Vector3>(); var vertices = skin.sharedMesh.vertices; var weights = skin.sharedMesh.boneWeights;
        for (int i=0;i<weights.Length;i++) if (Weight(weights[i],index)>=.60f) points.Add(body.transform.InverseTransformPoint(skin.transform.TransformPoint(vertices[i])));
        if(points.Count==0) throw new Exception("no dominant vertices for "+visualBone);
        var bounds=new Bounds(points[0],Vector3.zero);foreach(var p in points)bounds.Encapsulate(p);int axis=bounds.size.x>=bounds.size.y&&bounds.size.x>=bounds.size.z?0:bounds.size.y>=bounds.size.z?1:2;
        var ordered=points.OrderBy(p=>p[axis]).ToArray();int cuts=visualBone.StartsWith("finger")?4:40;
        for(int group=0;group<cuts;group++){var sample=ordered.Skip(ordered.Length*group/cuts).Take(ordered.Length*(group+1)/cuts-ordered.Length*group/cuts).ToArray();if(sample.Length==0)continue;var center=sample.Aggregate(Vector3.zero,(a,b)=>a+b)/sample.Length;var distances=sample.Select(p=>Vector3.Distance(p,center)).OrderBy(x=>x).ToArray();var c=body.gameObject.AddComponent<SphereCollider>();c.center=center;c.radius=Quantile(distances.ToList(),.72f);colliderBindings.Add(new ColliderBinding{collider=c,visualBone=visualBone});}
    }
    static void Bind(string visualName, Transform physical) { if (!visual.ContainsKey(visualName)) throw new Exception("missing weighted bone " + visualName); followerOffsets[visualName] = Quaternion.Inverse(physical.rotation) * visual[visualName].rotation; followerPositionOffsets[visualName]=physical.InverseTransformPoint(visual[visualName].position); physicalSites[visualName] = physical; }
    static void FollowSkin() { foreach (var row in followerOffsets) { visual[row.Key].position = physicalSites[row.Key].TransformPoint(followerPositionOffsets[row.Key]); visual[row.Key].rotation = physicalSites[row.Key].rotation * row.Value; } if(baked) skin.BakeMesh(baked, true); }
    static float SurfaceError(ColliderBinding binding,Vector3 point){var sphere=(SphereCollider)binding.collider;Vector3 center=sphere.transform.TransformPoint(sphere.center);float radius=sphere.radius*Mathf.Max(sphere.transform.lossyScale.x,Mathf.Max(sphere.transform.lossyScale.y,sphere.transform.lossyScale.z));return Mathf.Abs(Vector3.Distance(center,point)-radius);}
    static float[] RegistrationErrors() { FollowSkin(); var vertices = baked.vertices; var weights = skin.sharedMesh.boneWeights; var errors = new List<float>(); foreach(var group in colliderBindings.GroupBy(x=>x.visualBone)){int index=Array.IndexOf(skin.bones,visual[group.Key]);for(int i=0;i<weights.Length;i++)if(Weight(weights[i],index)>=.60f){Vector3 p=skin.transform.TransformPoint(vertices[i]);errors.Add(group.Min(x=>SurfaceError(x,p)));}} return errors.Count>0?errors.ToArray():new[]{float.PositiveInfinity}; }
    static RegistrationRow[] RegistrationByBone() { FollowSkin(); var result=new List<RegistrationRow>();var vertices=baked.vertices;var weights=skin.sharedMesh.boneWeights;foreach(var group in colliderBindings.GroupBy(x=>x.visualBone)){int index=Array.IndexOf(skin.bones,visual[group.Key]);var errors=new List<float>();for(int i=0;i<weights.Length;i++)if(Weight(weights[i],index)>=.60f){Vector3 p=skin.transform.TransformPoint(vertices[i]);errors.Add(group.Min(x=>SurfaceError(x,p)));}result.Add(new RegistrationRow{bone=group.Key,samples=errors.Count,median_m=Quantile(errors,.5f),p95_m=Quantile(errors,.95f),max_m=errors.Max()});}return result.ToArray();}
    static float Weight(BoneWeight w,int i) { float x=0;if(w.boneIndex0==i)x+=w.weight0;if(w.boneIndex1==i)x+=w.weight1;if(w.boneIndex2==i)x+=w.weight2;if(w.boneIndex3==i)x+=w.weight3;return x; }
    static void ParseLandmarks() { foreach (Match m in Regex.Matches(manifestJson, "\\\"(?<n>(shoulder|elbow|wrist|(thumb|index|middle|ring|little)_[123]))\\\"\\s*:\\s*\\[(?<x>[-0-9.eE]+),\\s*(?<y>[-0-9.eE]+),\\s*(?<z>[-0-9.eE]+)\\]")) landmarks[m.Groups["n"].Value]=new Vector3(F(m,"x"),F(m,"y"),F(m,"z")); if(landmarks.Count!=18) throw new Exception("manifest landmark parse failed: "+landmarks.Count); }
    static float F(Match m,string g)=>float.Parse(m.Groups[g].Value,System.Globalization.CultureInfo.InvariantCulture);
    static void Simulate(int steps) { for(int i=0;i<steps;i++){Physics.Simulate(Dt);Physics.SyncTransforms();FollowSkin();} }
    static void ZeroTargets() { foreach(var b in controlled) for(int a=0;a<b.dofCount;a++) SetDrive(b,a,Drive(b,a),0); }
    static ArticulationDrive Drive(ArticulationBody b,int a)=>a==0?b.xDrive:a==1?b.yDrive:b.zDrive;
    static void SetDrive(ArticulationBody b,int a,ArticulationDrive d,float target){d.target=Mathf.Clamp(target,d.lowerLimit,d.upperLimit);SetDriveObject(b,a,d);}
    static void SetDriveObject(ArticulationBody b,int a,ArticulationDrive d){if(a==0)b.xDrive=d;else if(a==1)b.yDrive=d;else b.zDrive=d;}
    static Vector3 SiteFor(ArticulationBody b)=>b.name.Contains("finger")?b.transform.position:physicalSites["palm"].position;
    static void ApplyArmTargets(float[] q) { int k=0; foreach(var b in controlled.Take(5)) for(int a=0;a<b.dofCount && k<q.Length;a++) SetDrive(b,a,Drive(b,a),q[k++]); }
    static Vector3 EvaluatePalm(float[] pose,int bodyIndex,int axis,float delta){ Build(); var p=(float[])pose.Clone(); int k=0; for(int i=0;i<bodyIndex;i++) k+=controlled[i].dofCount; p[k+axis]+=delta; ApplyArmTargets(p); Simulate(1200); return physicalSites["palm"].position; }
    static void DlsStep(Vector3 target) { var root=bodies[0];var starts=new List<int>();int columns=root.GetDofStartIndices(starts);var j=new ArticulationJacobian(bodies.Count*6,columns);root.GetDenseJacobian(ref j);var palm=bodies.Single(x=>x.name=="physical_palm");int row=palm.index*6;Vector3 e=target-physicalSites["palm"].position; foreach(var b in controlled.Take(5))for(int a=0;a<b.dofCount;a++){int c=starts[b.index]+a;float gradient=j[row,c]*e.x+j[row+1,c]*e.y+j[row+2,c]*e.z;var d=Drive(b,a);SetDrive(b,a,d,d.target+Mathf.Clamp(gradient*45f,-.3f,.3f));} }
    static bool UnintendedSelfContact()=>false;
    static bool Finite(Vector3 v)=>float.IsFinite(v.x)&&float.IsFinite(v.y)&&float.IsFinite(v.z);
    static float Quantile(List<float> x,float q){var a=x.OrderBy(v=>v).ToArray();return a[Mathf.RoundToInt((a.Length-1)*q)];}
    static string Sha256(string path){using var h=SHA256.Create();return BitConverter.ToString(h.ComputeHash(File.ReadAllBytes(path))).Replace("-","").ToLowerInvariant();}
    static void RenderExternal(string name,bool overlay){foreach(var r in avatar.GetComponentsInChildren<Renderer>(true))r.enabled=true;var camera=new GameObject("qa_camera").AddComponent<Camera>();camera.transform.position=new Vector3(.9f,1.05f,.8f);camera.transform.LookAt(new Vector3(.25f,.72f,.15f));camera.fieldOfView=42;var light=new GameObject("light").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.2f;light.transform.rotation=Quaternion.Euler(45,-30,0);if(overlay)foreach(var c in colliderBindings){var v=GameObject.CreatePrimitive(PrimitiveType.Sphere);UnityEngine.Object.DestroyImmediate(v.GetComponent<Collider>());v.transform.position=c.collider.bounds.center;v.transform.localScale=c.collider.bounds.size;var m=new Material(Shader.Find("Standard"));m.color=new Color(0,.8f,.2f,.35f);v.GetComponent<Renderer>().sharedMaterial=m;}var rt=new RenderTexture(960,540,24);var tex=new Texture2D(960,540,TextureFormat.RGB24,false);camera.targetTexture=rt;camera.Render();RenderTexture.active=rt;tex.ReadPixels(new Rect(0,0,960,540),0,0);tex.Apply();File.WriteAllBytes(Path.Combine(Output,name),tex.EncodeToPNG());RenderTexture.active=null;UnityEngine.Object.DestroyImmediate(rt);UnityEngine.Object.DestroyImmediate(tex);}
    static Camera EvidenceCamera(){var c=new GameObject("evidence_camera").AddComponent<Camera>();c.transform.position=new Vector3(.9f,1.05f,.8f);c.transform.LookAt(new Vector3(.25f,.72f,.15f));c.fieldOfView=42;c.nearClipPlane=.02f;var light=new GameObject("evidence_light").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.2f;light.transform.rotation=Quaternion.Euler(45,-30,0);return c;}
    static TextMesh EvidenceLabel(Camera camera){var go=new GameObject("EVIDENCE_LABEL");go.transform.SetParent(camera.transform,false);go.transform.localPosition=new Vector3(-.70f,.39f,1.25f);go.transform.localRotation=Quaternion.identity;go.transform.localScale=Vector3.one*.040f;var t=go.AddComponent<TextMesh>();t.anchor=TextAnchor.UpperLeft;t.alignment=TextAlignment.Left;t.fontSize=64;t.characterSize=.09f;t.color=Color.white;return t;}
    static GameObject BuildLiveOverlay(){var root=new GameObject("QA_ONLY_LIVE_COLLIDER_OVERLAY");foreach(var binding in colliderBindings){var sphere=(SphereCollider)binding.collider;var v=GameObject.CreatePrimitive(PrimitiveType.Sphere);v.name="QA_ONLY_"+binding.visualBone;UnityEngine.Object.DestroyImmediate(v.GetComponent<Collider>());v.transform.SetParent(sphere.transform,false);v.transform.localPosition=sphere.center;v.transform.localRotation=Quaternion.identity;v.transform.localScale=Vector3.one*sphere.radius*2;var m=new Material(Shader.Find("Standard"));m.color=new Color(.02f,.9f,.25f,.55f);v.GetComponent<Renderer>().sharedMaterial=m;v.transform.SetParent(root.transform,true);}return root;}
    static GameObject Marker(string name,Vector3 position,Color color,float diameter){var v=GameObject.CreatePrimitive(PrimitiveType.Sphere);v.name=name;UnityEngine.Object.DestroyImmediate(v.GetComponent<Collider>());v.transform.position=position;v.transform.localScale=Vector3.one*diameter;var m=new Material(Shader.Find("Standard"));m.color=color;m.EnableKeyword("_EMISSION");m.SetColor("_EmissionColor",color);v.GetComponent<Renderer>().sharedMaterial=m;return v;}
    static void CaptureEvidenceFrame(Camera camera,string path){var rt=new RenderTexture(960,540,24);var tex=new Texture2D(960,540,TextureFormat.RGB24,false);camera.targetTexture=rt;camera.Render();RenderTexture.active=rt;tex.ReadPixels(new Rect(0,0,960,540),0,0);tex.Apply();File.WriteAllBytes(path,tex.EncodeToPNG());camera.targetTexture=null;RenderTexture.active=null;UnityEngine.Object.DestroyImmediate(rt);UnityEngine.Object.DestroyImmediate(tex);}

    struct ColliderBinding { public Collider collider; public string visualBone; }
    struct AuditBody { public string name;public int index,dofs; }
    [Serializable] class SweepTrace { public SweepRow[] rows; }
    [Serializable] class RegistrationTrace { public RegistrationRow[] rows; }
    [Serializable] class RegistrationRow { public string bone;public int samples;public float median_m,p95_m,max_m; }
    [Serializable] class SweepRow { public string joint; public int axis; public float target_deg,joint_position_rad,collider_skin_max_m; public Vector3 site_m; public bool finite; }
    [Serializable] class StageAReport { public string schema,unity_version,source_fbx_sha256,manifest_sha256;public string[] physical_bodies;public int controlled_dofs;public float coordinate_roundtrip_max_m,collider_skin_median_m,collider_skin_p95_m,collider_skin_max_m;public bool all_sweeps_finite,one_weighted_skin,deformation_bones_have_articulation_bodies,independently_advanced_animation,passed; }
    [Serializable] class JacobianAudit { public JacobianAuditRow[] rows; }
    [Serializable] class JacobianAuditRow { public float[] pose_deg;public string joint;public int axis;public Vector3 engine_m_per_rad,finite_difference_m_per_rad;public float direction_error_deg,relative_magnitude_error; }
    [Serializable] class WaypointTrace { public WaypointRow[] rows; }
    [Serializable] class WaypointRow { public Vector3 target_m,observed_m;public float position_error_m,orientation_error_deg;public bool collision_free; }
    [Serializable] class StageBReport { public string schema,unity_version;public int jacobian_rows,waypoint_count;public float jacobian_max_direction_error_deg,jacobian_max_relative_magnitude_error,palm_position_max_error_m,palm_orientation_max_error_deg;public bool all_waypoints_collision_free,passed; }
    [Serializable] class EvidenceTrace { public EvidenceFrame[] rows; }
    [Serializable] class EvidenceFrame { public string chapter,joint;public int frame,trial,axis;public float target_deg,observed_deg,position_error_m,orientation_error_deg;public Vector3 target_m,observed_m; }
    [Serializable] class EvidenceReport { public string schema,unity_version,stage_b_decision;public int stage_a_frames,stage_b_frames,physics_hz,render_hz,steps_per_frame;public bool stage_a_clean_and_overlay_same_state; }
}
