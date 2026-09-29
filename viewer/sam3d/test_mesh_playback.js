const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(require('node:path').join(__dirname, 'mesh_renderer.js'), 'utf8');
const context = {devicePixelRatio: 1};
vm.runInNewContext(source + '\nthis.SamMeshRenderer = SamMeshRenderer;', context);

const renderer = Object.create(context.SamMeshRenderer.prototype);
renderer.ready = true;
renderer.video = {readyState: 4};
renderer.meta = {vertices: 1, frames: 250};
renderer.meshFrames = new Map();
renderer.meshWhole = new Map();
renderer.meshPending = new Map();
renderer.canvas = {width: 100, height: 100};
renderer.renderer={clear(){throw Error('buffering must preserve the displayed frame')}};
const requested = [];
renderer.frameMesh = (file, n) => {requested.push(n); return Promise.resolve();};

const drawn = renderer.draw(7, 'raw', [], [], [], [], 1, {width: 100, height: 100});
assert.equal(drawn, false, 'missing mesh must buffer without clearing the displayed frame');
assert.ok(requested.includes(7), 'current frame must be requested');
assert.ok(requested.includes(8), 'upcoming frames must be prefetched');
console.log('mesh playback retained frame and prefetch OK');
