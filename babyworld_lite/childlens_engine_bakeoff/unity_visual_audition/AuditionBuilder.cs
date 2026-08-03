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
    static GameObject avatar;
    static Transform root, head, wrist;
    static readonly List<Transform> ik = new List<Transform>();
    static readonly List<Transform> fingers = new List<Transform>();
    static readonly Dictionary<Transform, Quaternion> rest = new Dictionary<Transform, Quaternion>();
    static Camera camera;
    static Vector3 touchPoint;

    [MenuItem("BabyWorld/Render Visual Audition")]
    public static void Render()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("UNITY_AUDITION_OUTPUT is required");
        Directory.CreateDirectory(Output);
        BuildScene();
        var rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
        var tex = new Texture2D(Width, Height, TextureFormat.RGB24, false);
        camera.targetTexture = rt;
        for (int frame = 0; frame < Frames; frame++) {
            Pose(frame / (float)Fps);
            camera.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, Width, Height), 0, 0); tex.Apply(false);
            File.WriteAllBytes(Path.Combine(Output, $"frame_{frame:D4}.png"), tex.EncodeToPNG());
        }
        var bounds = avatar.GetComponentsInChildren<Renderer>().Select(r => r.bounds).Aggregate((a,b) => { a.Encapsulate(b); return a; });
        var closest = bounds.ClosestPoint(camera.transform.position);
        File.WriteAllText(Path.Combine(Output, "camera_mount.json"), JsonUtility.ToJson(new MountRecord {
            editor="6000.0.80f1", pipeline="Built-in Render Pipeline", resolution="960x540", fps=Fps,
            mount_local_position=camera.transform.localPosition, mount_local_euler=camera.transform.localEulerAngles,
            head_world_position=head.position, camera_world_position=camera.transform.position, avatar_bounds_size=bounds.size,
            camera_outside_avatar_bounds=!bounds.Contains(camera.transform.position), aabb_clearance_m=Vector3.Distance(closest,camera.transform.position),
            frozen_head_radius_m=.16f, documented_head_surface_clearance_m=Vector3.Distance(head.position,camera.transform.position)-.16f,
            vertical_fov_deg=camera.fieldOfView, near_clip_m=camera.nearClipPlane,
            disclosure="All motion, IK, finger closure, and touch are kinematic/nonphysical; target is static. Visual feasibility only."
        }, true));
        AssetDatabase.SaveAssets(); EditorApplication.Exit(0);
    }

    static void BuildScene()
    {
        foreach (var o in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) UnityEngine.Object.DestroyImmediate(o);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(.48f,.57f,.68f); RenderSettings.ambientEquatorColor = new Color(.28f,.25f,.23f); RenderSettings.ambientGroundColor = new Color(.12f,.10f,.09f);
        RenderSettings.fog = true; RenderSettings.fogColor = new Color(.62f,.69f,.74f); RenderSettings.fogDensity=.012f;
        Mat("Floor", new Color(.42f,.28f,.17f)); Box("Floor", new Vector3(0,-.035f,-1.7f), new Vector3(4.8f,.07f,5.2f), "Floor");
        Mat("Wall", new Color(.70f,.76f,.72f)); Box("BackWall", new Vector3(0,1.35f,-4.25f), new Vector3(4.8f,2.7f,.08f), "Wall"); Box("SideWall", new Vector3(-2.36f,1.35f,-1.7f), new Vector3(.08f,2.7f,5.2f), "Wall");
        Mat("Trim", new Color(.91f,.88f,.78f)); Box("Baseboard",new Vector3(0,.09f,-4.16f),new Vector3(4.7f,.18f,.08f),"Trim");
        Place("rugRectangle", new Vector3(.15f,.012f,-1.65f), new Vector3(2.1f,1f,2.1f), 0);
        Place("loungeSofaLong", new Vector3(1.05f,0,-3.55f), Vector3.one*1.35f, 0);
        Place("bookcaseOpen", new Vector3(-1.75f,0,-3.55f), Vector3.one*1.35f, 0);
        Place("pottedPlant", new Vector3(-1.62f,0,-2.65f), Vector3.one*1.4f, 180);
        Place("lampSquareFloor", new Vector3(1.92f,0,-3.4f), Vector3.one*1.25f, 180);
        Place("tableCoffee", new Vector3(.25f,0,-1.55f), Vector3.one*1.4f, 180);
        Place("books", new Vector3(.45f,.49f,-1.55f), Vector3.one*1.2f, 192);
        Place("chairCushion", new Vector3(-1.25f,0,-1.25f), Vector3.one*1.25f, 205);
        Mat("Toy", new Color(.86f,.28f,.16f)); var toy=GameObject.CreatePrimitive(PrimitiveType.Sphere); toy.name="NearbyRedToy"; toy.transform.position=new Vector3(-.24f,.60f,-1.28f); toy.transform.localScale=new Vector3(.16f,.16f,.16f); toy.GetComponent<Renderer>().sharedMaterial=AssetDatabase.LoadAssetAtPath<Material>("Assets/Generated/Toy.mat"); touchPoint=toy.transform.position + new Vector3(0,0,-.08f);
        avatar = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Avatar/child.fbx")); avatar.name="CC0_Child_Avatar"; avatar.transform.position=new Vector3(0,0,0); avatar.transform.rotation=Quaternion.Euler(0,0,0); avatar.transform.localScale=Vector3.one*1.9f;
        foreach(var r in avatar.GetComponentsInChildren<Renderer>()) { var m=new Material(Shader.Find("Standard")); m.color=new Color(.74f,.49f,.36f); m.SetFloat("_Glossiness",.22f); r.sharedMaterial=m; }
        root=avatar.transform; head=Find("head"); wrist=Find("wrist.R");
        foreach(var n in new[]{"upperarm01.R","upperarm02.R","lowerarm01.R","lowerarm02.R","wrist.R"}) ik.Add(Find(n));
        foreach(var t in avatar.GetComponentsInChildren<Transform>()) if(t.name.StartsWith("finger") && t.name.EndsWith(".R")) fingers.Add(t);
        foreach(var t in avatar.GetComponentsInChildren<Transform>()) rest[t]=t.localRotation;
        var camGo=new GameObject("FrozenHeadCameraMount"); camGo.transform.SetParent(head,false); camGo.transform.localPosition=new Vector3(0,.00020f,-.00250f); camGo.transform.localRotation=Quaternion.Euler(8,180,0);
        camera=camGo.AddComponent<Camera>(); camera.fieldOfView=58; camera.nearClipPlane=.025f; camera.farClipPlane=30; camera.clearFlags=CameraClearFlags.Skybox;
        var sun=new GameObject("WindowKey").AddComponent<Light>(); sun.type=LightType.Directional; sun.intensity=1.15f; sun.color=new Color(1f,.88f,.72f); sun.transform.rotation=Quaternion.Euler(42,-28,0); sun.shadows=LightShadows.Soft;
        var fill=new GameObject("WarmFill").AddComponent<Light>(); fill.type=LightType.Point; fill.range=7; fill.intensity=4.2f; fill.color=new Color(1f,.72f,.48f); fill.transform.position=new Vector3(1.6f,2.2f,1.5f); fill.shadows=LightShadows.Soft;
    }

    static void Pose(float t)
    {
        foreach(var kv in rest) if(kv.Key) kv.Key.localRotation=kv.Value;
        float look = Smooth(0,1.55f,t) - Smooth(1.55f,2.35f,t);
        head.localRotation = rest[head] * Quaternion.Euler(9f*look,-16f*look,2f*look);
        float reach = Smooth(1.45f,3.65f,t); float withdraw = Smooth(4.65f,6.55f,t); float amount=reach*(1-withdraw);
        Vector3 target=Vector3.Lerp(wrist.position,touchPoint,amount);
        for(int pass=0;pass<12;pass++) for(int i=ik.Count-2;i>=0;i--) { var b=ik[i]; var delta=Quaternion.FromToRotation(wrist.position-b.position,target-b.position); b.rotation=Quaternion.Slerp(Quaternion.identity,delta,.62f)*b.rotation; }
        float close=Smooth(3.15f,3.9f,t)*(1-Smooth(4.45f,5.1f,t));
        foreach(var f in fingers) { int segment=f.name.Contains("-1.")?18:f.name.Contains("-2.")?28:34; f.localRotation=rest[f]*Quaternion.Euler(segment*close,0,0); }
    }
    static float Smooth(float a,float b,float t){ return Mathf.SmoothStep(0,1,Mathf.InverseLerp(a,b,t)); }
    static Transform Find(string n){ var t=avatar.GetComponentsInChildren<Transform>(true).FirstOrDefault(x=>x.name==n); if(!t) throw new Exception("Missing bone "+n); return t; }
    static void Place(string name,Vector3 p,Vector3 s,float yaw){ var prefab=AssetDatabase.LoadAssetAtPath<GameObject>($"Assets/Furniture/{name}.obj"); if(!prefab) throw new Exception("Missing furniture "+name); var o=(GameObject)PrefabUtility.InstantiatePrefab(prefab); o.name=name; o.transform.position=p; o.transform.localScale=s; o.transform.rotation=Quaternion.Euler(0,yaw,0); }
    static void Mat(string name,Color color){ Directory.CreateDirectory("Assets/Generated"); var path=$"Assets/Generated/{name}.mat"; var m=AssetDatabase.LoadAssetAtPath<Material>(path); if(!m){m=new Material(Shader.Find("Standard")); AssetDatabase.CreateAsset(m,path);} m.color=color; m.SetFloat("_Glossiness",.28f); }
    static void Box(string n,Vector3 p,Vector3 s,string mat){var o=GameObject.CreatePrimitive(PrimitiveType.Cube);o.name=n;o.transform.position=p;o.transform.localScale=s;o.GetComponent<Renderer>().sharedMaterial=AssetDatabase.LoadAssetAtPath<Material>($"Assets/Generated/{mat}.mat");}
    [Serializable] class MountRecord { public string editor,pipeline,resolution,disclosure; public int fps; public Vector3 mount_local_position,mount_local_euler,head_world_position,camera_world_position,avatar_bounds_size; public bool camera_outside_avatar_bounds; public float aabb_clearance_m,frozen_head_radius_m,documented_head_surface_clearance_m,vertical_fov_deg,near_clip_m; }
}
