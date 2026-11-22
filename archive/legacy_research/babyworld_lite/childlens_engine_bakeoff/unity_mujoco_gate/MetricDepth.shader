Shader "BabyWorld/MetricDepth" {
  SubShader { Tags { "RenderType"="Opaque" }
    Pass { CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag
      #include "UnityCG.cginc"
      struct v2f { float4 pos : SV_POSITION; float depth : TEXCOORD0; };
      v2f vert(appdata_base v) { v2f o; o.pos = UnityObjectToClipPos(v.vertex); o.depth = -UnityObjectToViewPos(v.vertex).z; return o; }
      fixed4 frag(v2f i) : SV_Target { float d = saturate(i.depth / 5.0); return fixed4(d, d, d, 1); }
    ENDCG }
  }
}
