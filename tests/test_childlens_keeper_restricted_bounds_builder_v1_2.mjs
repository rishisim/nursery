import assert from "node:assert/strict";
import test from "node:test";
import { childLensKeeperRestrictedBoundsBuilderV12 } from "../scripts/childlens_keeper_restricted_bounds_builder_v1_2.mjs";

const LIBRARY = "c8ed0104-b793-4c35-817e-302afd4e036b";
const ROW_SELECTOR = "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]";
const SIZE_SELECTOR = "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']";
function node({attributes={},text="",one={},many={}}={}) { return {getAttribute:k=>Object.hasOwn(attributes,k)?attributes[k]:null,get textContent(){return text;},querySelector:s=>one[s]??null,querySelectorAll:s=>many[s]??[]}; }
function row(name,size) { return node({attributes:{"data-file-name":name},one:{[SIZE_SELECTOR]:node({text:size})}}); }
function fixture(rows) { return node({many:{[ROW_SELECTOR]:rows,"a[href]":[]}}); }
function withPage(doc, fn) { const od=globalThis.document,ol=globalThis.location; globalThis.document=doc; globalThis.location={href:`https://keeper.invalid/library/${LIBRARY}/`}; try{return fn();} finally{if(od===undefined)delete globalThis.document;else globalThis.document=od;if(ol===undefined)delete globalThis.location;else globalThis.location=ol;} }

test("returns numeric bounds in frozen order without source labels",()=>{
  const a="PRIVATE_ALPHA.mp4",b="PRIVATE_BETA.mp4";
  const out=withPage(fixture([row(b,"800 MB"),row(a,"1.2 GB")]),()=>childLensKeeperRestrictedBoundsBuilderV12({sourceLocators:[`/ChildLens/videos/${a}`,`/ChildLens/videos/${b}`]}));
  assert.equal(out.status,"ok"); assert.deepEqual(out.bounds,[{displayBytes:1_200_000_000,quantumBytes:100_000_000},{displayBytes:800_000_000,quantumBytes:1_000_000}]);
  const serialized=JSON.stringify(out); assert.equal(serialized.includes(a),false); assert.equal(serialized.includes(b),false);
});
test("missing and duplicate rows fail closed",()=>{
  const missing=withPage(fixture([]),()=>childLensKeeperRestrictedBoundsBuilderV12({sourceLocators:["/ChildLens/videos/a.mp4"]}));
  const duplicate=withPage(fixture([row("a.mp4","1 GB"),row("a.mp4","1 GB")]),()=>childLensKeeperRestrictedBoundsBuilderV12({sourceLocators:["/ChildLens/videos/a.mp4"]}));
  assert.equal(missing.errorCode,"E_ROWS"); assert.equal(duplicate.errorCode,"E_DUPLICATE"); assert.deepEqual(duplicate.bounds,[]);
});
test("invalid source and wrong identity fail closed",()=>{
  assert.equal(childLensKeeperRestrictedBoundsBuilderV12({sourceLocators:["https://invalid/a.mp4"]}).errorCode,"E_CONFIG");
  const od=globalThis.document,ol=globalThis.location;globalThis.document=fixture([row("a.mp4","1 GB")]);globalThis.location={href:"https://invalid/"};try{assert.equal(childLensKeeperRestrictedBoundsBuilderV12({sourceLocators:["/ChildLens/videos/a.mp4"]}).errorCode,"E_IDENTITY");}finally{if(od===undefined)delete globalThis.document;else globalThis.document=od;if(ol===undefined)delete globalThis.location;else globalThis.location=ol;}
});
test("serialized function has no closure dependency",()=>{
  const fn=Function(`return (${childLensKeeperRestrictedBoundsBuilderV12.toString()});`)();
  const out=withPage(fixture([row("a.mp4","1 GB")]),()=>fn({sourceLocators:["/ChildLens/videos/a.mp4"]})); assert.equal(out.status,"ok");
});
