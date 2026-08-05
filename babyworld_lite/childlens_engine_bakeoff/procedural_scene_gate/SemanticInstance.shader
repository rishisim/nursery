Shader "ProceduralSceneGate/SemanticInstanceUint24"
{
    Properties
    {
        _SemanticId ("Semantic uint24", Float) = 0
        _InstanceId ("Persistent instance uint24", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float _SemanticId;
            float _InstanceId;
            float _RegisteredLabelMode;

            struct v2f { float4 position : SV_POSITION; };

            v2f vert(appdata_base input)
            {
                v2f output;
                output.position = UnityObjectToClipPos(input.vertex);
                return output;
            }

            float4 frag(v2f input) : SV_Target
            {
                float identifier = clamp(floor(lerp(_SemanticId, _InstanceId, step(0.5, _RegisteredLabelMode)) + 0.5), 0.0, 16777215.0);
                float red = fmod(identifier, 256.0);
                float green = fmod(floor(identifier / 256.0), 256.0);
                float blue = floor(identifier / 65536.0);
                return float4(float3(red, green, blue) / 255.0, 1.0);
            }
            ENDCG
        }
    }
    Fallback Off
}
