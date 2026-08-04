using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Hard Stage-B cell: kinematically actuated visible hands; one free PhysX object.
// No parenting, joints, object transform writes after initialization, forces,
// springs, trajectories, catch surfaces, or post-physics repair are present.
public static class HybridBimanualCellBuilder
{
    const float Dt=1f/240f;
    static int Steps=2400;
    static readonly string Output=Environment.GetEnvironmentVariable("HYBRID_CELL_OUTPUT");
    static readonly List<ContactEvent> Contacts=new();
    static readonly List<ColliderBinding> HandColliders=new();
    static Rigidbody target,blueCup,rightBody,leftBody;
    static Transform right,left;
    static readonly List<Digit> rightDigits=new(),leftDigits=new();
    static Vector3 initialTarget;
    static Quaternion turnReference;
    static float maxHeight,minPenetration=0,maxPalmSpeed,maxPalmAngularSpeed,maxClosureRate,rotationDeg;
    static bool released;
    static bool fullEpisode;static Vector3 blueInitial;
    static float targetLateral,targetMass=.035f,targetFriction=1f;

    [MenuItem("BabyWorld/Run Hybrid Bimanual Hard Cell")]
    public static void Run()
    {
        if(string.IsNullOrWhiteSpace(Output))throw new Exception("HYBRID_CELL_OUTPUT is required");
        fullEpisode=Environment.GetEnvironmentVariable("HYBRID_FULL_EPISODE")=="1";targetLateral=EnvFloat("HYBRID_TARGET_LATERAL_M",0);targetMass=EnvFloat("HYBRID_TARGET_MASS_KG",.035f);targetFriction=EnvFloat("HYBRID_TARGET_FRICTION",1);Steps=fullEpisode?56*240:2400;Directory.CreateDirectory(Output);Directory.CreateDirectory(Path.Combine(Output,"frames"));
        Build();var camera=BuildCamera();var rows=new List<TraceRow>();
        Vector3 priorRight=right.position,priorLeft=left.position;Quaternion priorRightRot=right.rotation,priorLeftRot=left.rotation;
        for(int step=0;step<=Steps;step++){
            float t=step*Dt;string phase=fullEpisode?DriveFull(t):Drive(t);
            Physics.SyncTransforms();Physics.Simulate(Dt);Physics.SyncTransforms();MeasureContacts(step);
            float rs=Vector3.Distance(right.position,priorRight)/Dt,ls=Vector3.Distance(left.position,priorLeft)/Dt;
            float ra=Quaternion.Angle(right.rotation,priorRightRot)/Dt,la=Quaternion.Angle(left.rotation,priorLeftRot)/Dt;
            maxPalmSpeed=Mathf.Max(maxPalmSpeed,rs,ls);maxPalmAngularSpeed=Mathf.Max(maxPalmAngularSpeed,ra,la);
            priorRight=right.position;priorLeft=left.position;priorRightRot=right.rotation;priorLeftRot=left.rotation;
            maxHeight=Mathf.Max(maxHeight,target.position.y-initialTarget.y);float turnStart=fullEpisode?6+5.2f*3.3f:5.2f,turnEnd=fullEpisode?6+6.5f*3.3f:6.5f;if(t>=turnStart&&t<turnStart+Dt*1.1f)turnReference=target.rotation;if(t>=turnStart&&t<=turnEnd)rotationDeg=Mathf.Max(rotationDeg,Quaternion.Angle(turnReference,target.rotation));
            if(step%8==0){rows.Add(new TraceRow{time_s=t,phase=phase,object_position_m=target.position,object_rotation=target.rotation,object_velocity_m_s=target.linearVelocity,blue_object_position_m=blueCup?blueCup.position:Vector3.zero,blue_object_rotation=blueCup?blueCup.rotation:Quaternion.identity,blue_object_velocity_m_s=blueCup?blueCup.linearVelocity:Vector3.zero,
                right_palm_position_m=right.position,left_palm_position_m=left.position,right_closure=Closure(rightDigits),left_closure=Closure(leftDigits),
                measured_contact_count=Contacts.Count(x=>x.step>=step-7),object_sleeping=target.IsSleeping()});if(!fullEpisode)Capture(camera,Path.Combine(Output,"frames",$"frame_{step/8:D4}.png"));}
        }
        float settleSpeed=target.linearVelocity.magnitude;bool rightContact=Contacts.Any(x=>x.hand=="right"),leftContact=Contacts.Any(x=>x.hand=="left");
        var report=new Report{schema="embodied.hybrid_bimanual_hard_cell.v1",unity_version=Application.unityVersion,physics_hz=240,render_hz=30,steps_per_frame=8,
            controller="kinematic palms and contact-aware kinematic digits; free non-kinematic PhysX object",object_pose_writes_after_initialization=0,object_external_forces=0,
            attachment_or_joint_count=0,assistance_ledger_entries=0,right_measured_contact=rightContact,left_measured_contact=leftContact,
            lift_m=maxHeight,turn_deg=rotationDeg,release_commanded=released,final_object_speed_m_s=settleSpeed,max_finger_penetration_m=-minPenetration,
            max_palm_speed_m_s=maxPalmSpeed,max_palm_angular_speed_deg_s=maxPalmAngularSpeed,max_closure_rate_s=maxClosureRate,available_impulse_status="UNAVAILABLE",duration_s=Steps*Dt,blue_cup_displacement_m=blueCup?Vector3.Distance(blueInitial,blueCup.position):0,target_lateral_m=targetLateral,target_mass_kg=targetMass,target_friction=targetFriction};
        report.passed=rightContact&&leftContact&&maxHeight>=.08f&&rotationDeg>=20&&released&&settleSpeed<.08f&&-minPenetration<=.003f&&maxPalmSpeed<=.18f&&maxPalmAngularSpeed<=45&&maxClosureRate<=3.1f&&report.assistance_ledger_entries==0&&(!fullEpisode||report.blue_cup_displacement_m>=.01f);
        File.WriteAllText(Path.Combine(Output,"trace.json"),JsonUtility.ToJson(new Trace{rows=rows.ToArray()},true));
        File.WriteAllText(Path.Combine(Output,"contacts.json"),JsonUtility.ToJson(new ContactTrace{rows=Contacts.ToArray()},true));
        File.WriteAllText(Path.Combine(Output,"report.json"),JsonUtility.ToJson(report,true));EditorApplication.Exit(report.passed?0:2);
    }

    static void Build(){Physics.simulationMode=SimulationMode.Script;Physics.gravity=new Vector3(0,-9.81f,0);Physics.defaultSolverIterations=24;Physics.defaultSolverVelocityIterations=12;
        var support=Cube("support",new Vector3(0,.0125f,0),new Vector3(.42f,.025f,.35f),new Color(.62f,.43f,.24f));
        target=Cube("free_red_toy",new Vector3(targetLateral,.053f,0),Vector3.one*.055f,new Color(.85f,.035f,.025f)).AddComponent<Rigidbody>();target.mass=targetMass;target.interpolation=RigidbodyInterpolation.None;target.collisionDetectionMode=CollisionDetectionMode.ContinuousDynamic;target.maxDepenetrationVelocity=.25f;
        var pm=new PhysicsMaterial("target_friction"){staticFriction=targetFriction,dynamicFriction=targetFriction*.9f,frictionCombine=PhysicsMaterialCombine.Maximum,bounciness=0};target.GetComponent<Collider>().material=pm;initialTarget=target.position;
        right=Hand("right",new Vector3(0,.066f,-.20f),new Color(.78f,.49f,.34f),out rightBody,rightDigits);
        left=Hand("left",new Vector3(-.18f,.15f,.01f),new Color(.72f,.43f,.30f),out leftBody,leftDigits);
        if(fullEpisode){blueCup=Cube("free_blue_cup",new Vector3(.14f,.063f,-.05f),new Vector3(.06f,.075f,.06f),new Color(.04f,.18f,.85f)).AddComponent<Rigidbody>();blueCup.mass=.05f;blueCup.collisionDetectionMode=CollisionDetectionMode.ContinuousDynamic;blueCup.GetComponent<Collider>().material=pm;blueInitial=blueCup.position;}
        var light=new GameObject("key_light").AddComponent<Light>();light.type=LightType.Directional;light.intensity=1.5f;light.transform.rotation=Quaternion.Euler(48,-28,0);RenderSettings.ambientLight=new Color(.32f,.30f,.28f);Physics.SyncTransforms();}
    static Transform Hand(string side,Vector3 position,Color color,out Rigidbody rb,List<Digit> digits){var root=new GameObject(side+"_hand_root").transform;root.position=position;rb=root.gameObject.AddComponent<Rigidbody>();rb.isKinematic=true;rb.collisionDetectionMode=CollisionDetectionMode.ContinuousSpeculative;var palm=Cube(side+"_palm",Vector3.zero,new Vector3(.07f,.075f,.018f),color);palm.transform.SetParent(root,false);HandColliders.Add(new ColliderBinding{collider=palm.GetComponent<Collider>(),hand=side,digit="palm"});
        string[] names={"thumb","index","middle","ring","little"};float[] ys={-.027f,.029f,.014f,-.003f,-.021f};float[] xs={-.046f,.046f,.048f,.048f,.045f};
        for(int d=0;d<5;d++){var digit=new Digit{name=names[d],side=side,openX=xs[d],closedX=Mathf.Sign(xs[d])*.034f};for(int p=0;p<3;p++){var seg=Cube(side+"_"+names[d]+"_"+(p+1),Vector3.zero,new Vector3(.015f,.015f,.030f),color);seg.transform.SetParent(root,false);seg.transform.localPosition=new Vector3(xs[d],ys[d],.018f+p*.027f);HandColliders.Add(new ColliderBinding{collider=seg.GetComponent<Collider>(),hand=side,digit=names[d]});digit.segments.Add(seg.transform);}digits.Add(digit);}return root;}
    static string Drive(float t){Vector3 r0=new(0,.066f,-.20f),rContact=new(0,.066f,-.0345f),rLift=new(0,.166f,-.0345f),rPlace=new(-.09f,.070f,.08f);Vector3 l0=new(-.18f,.15f,.01f),lAssist=new(-.075f,.166f,-.005f);Vector3 leftPlace=rPlace+new Vector3(-.084f,0,.03f);
        string phase;Vector3 rp,lp;Quaternion rr=Quaternion.identity,lr=Quaternion.identity;float rc=0,lc=0;
        if(t<1.5f){phase="right_reach";rp=Vector3.Lerp(r0,rContact,Smooth(t/1.5f));lp=l0;}
        else if(t<2.4f){phase="contact_aware_capture";rp=rContact;lp=l0;rc=Smooth((t-1.5f)/.9f);}
        else if(t<4.0f){phase="lift";rp=Vector3.Lerp(rContact,rLift,Smooth((t-2.4f)/1.6f));lp=l0;rc=1;}
        else if(t<5.2f){phase="left_assist";rp=rLift;lp=Vector3.Lerp(l0,lAssist,Smooth((t-4)/1.2f));rc=1;lc=Smooth((t-4.7f)/.5f);}
        else if(t<6.5f){phase="bimanual_turn";float a=25*Smooth((t-5.2f)/1.3f);rr=Quaternion.Euler(0,0,a);lr=rr;rp=rLift;lp=lAssist;rc=lc=1;}
        else if(t<8.0f){phase="lower_place";float u=Smooth((t-6.5f)/1.5f);rr=Quaternion.Euler(0,0,25*(1-u));lr=rr;rp=Vector3.Lerp(rLift,rPlace,u);lp=Vector3.Lerp(lAssist,leftPlace,u);rc=1;lc=1-u;}
        else if(t<8.7f){phase="commanded_open_release";rp=rPlace;lp=leftPlace;rc=1-Smooth((t-8)/.7f);released=true;}
        else{phase=t<9.4f?"free_settle":"withdrawal";float u=Smooth((t-8.7f)/1.3f);rp=Vector3.Lerp(rPlace,new Vector3(-.09f,.11f,-.04f),u);lp=Vector3.Lerp(leftPlace,new Vector3(-.20f,.13f,.00f),u);}
        Move(rightBody,rp,rr);Move(leftBody,lp,lr);SetClosure(rightDigits,rc);SetClosure(leftDigits,lc);return phase;}
    static string DriveFull(float t){if(t<6){Move(rightBody,new Vector3(0,.066f,-.20f),Quaternion.identity);Move(leftBody,new Vector3(-.18f,.15f,.01f),Quaternion.identity);SetClosure(rightDigits,0);SetClosure(leftDigits,0);return t<3?"scan_reorient":"gaze_red_toy";}if(t<39)return Drive((t-6)/3.3f);Vector3 leftEnd=new(-.20f,.13f,0),leftRest=new(-.18f,.14f,-.12f);Move(leftBody,t<41?Vector3.Lerp(leftEnd,leftRest,Smooth((t-39)/2)):leftRest,Quaternion.identity);SetClosure(leftDigits,0);Vector3 a=new(-.09f,.11f,-.25f),b=new(.14f,.11f,-.25f),c=new(.14f,.075f,-.085f),d=new(.14f,.075f,-.065f),e=new(.06f,.14f,-.20f);Vector3 rp;string phase;if(t<41){phase="withdraw_from_red";rp=Vector3.Lerp(new Vector3(-.09f,.11f,-.04f),a,Smooth((t-39)/2));}else if(t<43){phase="gaze_blue_cup_reach";rp=Vector3.Lerp(a,b,Smooth((t-41)/2));}else if(t<45){phase="blue_cup_touch";rp=Vector3.Lerp(b,c,Smooth((t-43)/2));}else if(t<46){phase="blue_cup_touch_push";rp=Vector3.Lerp(c,d,Smooth(t-45));}else if(t<48){phase="blue_cup_withdraw";rp=Vector3.Lerp(d,e,Smooth((t-46)/2));}else{phase=t<52?"final_withdrawal":"look_window_settle";rp=e;}Move(rightBody,rp,Quaternion.identity);SetClosure(rightDigits,0);return phase;}
    static void Move(Rigidbody rb,Vector3 p,Quaternion q){rb.MovePosition(p);rb.MoveRotation(q);}
    static void SetClosure(List<Digit> digits,float value){value=Mathf.Clamp01(value);foreach(var d in digits){float prior=d.value;if(value<prior){d.value=value;d.touchingTarget=false;}else if(!d.touchingTarget)d.value=value;maxClosureRate=Mathf.Max(maxClosureRate,Mathf.Abs(d.value-prior)/Dt);float x=Mathf.Lerp(d.openX,d.closedX,d.value);foreach(var s in d.segments){var p=s.localPosition;p.x=x;s.localPosition=p;}}}
    static float Closure(List<Digit> digits)=>digits.Average(x=>x.value);
    static void MeasureContacts(int step){var objectCollider=target.GetComponent<Collider>();foreach(var b in HandColliders){if(Physics.ComputePenetration(b.collider,b.collider.transform.position,b.collider.transform.rotation,objectCollider,objectCollider.transform.position,objectCollider.transform.rotation,out Vector3 direction,out float distance)){minPenetration=Mathf.Min(minPenetration,-distance);var list=b.hand=="right"?rightDigits:leftDigits;var d=list.FirstOrDefault(x=>x.name==b.digit);if(d!=null)d.touchingTarget=true;Contacts.Add(new ContactEvent{step=step,time_s=step*Dt,hand=b.hand,digit=b.digit,point_m=b.collider.ClosestPoint(target.position),normal=direction,separation_m=-distance});}}}
    static Camera BuildCamera(){var c=new GameObject("qa_camera").AddComponent<Camera>();c.transform.position=new Vector3(.58f,.42f,-.58f);c.transform.LookAt(new Vector3(0,.09f,.02f));c.fieldOfView=42;c.nearClipPlane=.02f;c.backgroundColor=new Color(.12f,.15f,.18f);return c;}
    static GameObject Cube(string name,Vector3 position,Vector3 scale,Color color){var go=GameObject.CreatePrimitive(PrimitiveType.Cube);go.name=name;go.transform.position=position;go.transform.localScale=scale;var mat=new Material(Shader.Find("Standard"));mat.color=color;go.GetComponent<Renderer>().sharedMaterial=mat;return go;}
    static void Capture(Camera c,string path){var rt=new RenderTexture(960,540,24);var tex=new Texture2D(960,540,TextureFormat.RGB24,false);c.targetTexture=rt;c.Render();RenderTexture.active=rt;tex.ReadPixels(new Rect(0,0,960,540),0,0);tex.Apply();File.WriteAllBytes(path,tex.EncodeToPNG());c.targetTexture=null;RenderTexture.active=null;UnityEngine.Object.DestroyImmediate(rt);UnityEngine.Object.DestroyImmediate(tex);}
    static float Smooth(float x){x=Mathf.Clamp01(x);return x*x*(3-2*x);}
    static float EnvFloat(string name,float fallback){return float.TryParse(Environment.GetEnvironmentVariable(name),System.Globalization.NumberStyles.Float,System.Globalization.CultureInfo.InvariantCulture,out float value)?value:fallback;}
    class Digit{public string name,side;public float openX,closedX,value;public bool touchingTarget;public readonly List<Transform> segments=new();}class ColliderBinding{public Collider collider;public string hand,digit;}
    [Serializable]class Trace{public TraceRow[] rows;}[Serializable]class ContactTrace{public ContactEvent[] rows;}
    [Serializable]class TraceRow{public float time_s;public string phase;public Vector3 object_position_m,blue_object_position_m;public Quaternion object_rotation,blue_object_rotation;public Vector3 object_velocity_m_s,blue_object_velocity_m_s,right_palm_position_m,left_palm_position_m;public float right_closure,left_closure;public int measured_contact_count;public bool object_sleeping;}
    [Serializable]public class ContactEvent{public int step;public float time_s;public string hand,digit;public Vector3 point_m,normal;public float separation_m;}
    [Serializable]class Report{public string schema,unity_version,controller,available_impulse_status;public int physics_hz,render_hz,steps_per_frame,object_pose_writes_after_initialization,object_external_forces,attachment_or_joint_count,assistance_ledger_entries;public bool right_measured_contact,left_measured_contact,release_commanded,passed;public float duration_s,target_lateral_m,target_mass_kg,target_friction,lift_m,turn_deg,blue_cup_displacement_m,final_object_speed_m_s,max_finger_penetration_m,max_palm_speed_m_s,max_palm_angular_speed_deg_s,max_closure_rate_s;}
}
