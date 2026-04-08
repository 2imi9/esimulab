// Wind particle advection compute shader (WebGPU)
// Advects particles through a 3D wind velocity field

struct Particle {
  position: vec3<f32>,
  lifetime: f32,
  velocity: vec3<f32>,
  age: f32,
};

struct SimParams {
  deltaTime: f32,
  windScale: f32,
  numParticles: u32,
  domainSize: vec3<f32>,
  _padding: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var windField: texture_3d<f32>;
@group(0) @binding(3) var windSampler: sampler;

// Map world position to wind field UVW coordinates
fn worldToUVW(pos: vec3<f32>) -> vec3<f32> {
  return (pos + params.domainSize * 0.5) / params.domainSize;
}

// Simple pseudo-random hash for particle respawn jitter
fn hash(seed: u32) -> f32 {
  var s = seed;
  s = s ^ (s >> 16u);
  s = s * 0x45d9f3bu;
  s = s ^ (s >> 16u);
  return f32(s & 0xFFFFu) / 65535.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= params.numParticles) {
    return;
  }

  var p = particles[idx];

  // Sample wind field at particle position
  let uvw = worldToUVW(p.position);
  let wind = textureSampleLevel(windField, windSampler, uvw, 0.0).xyz;

  // Advect particle
  p.velocity = wind * params.windScale;
  p.position += p.velocity * params.deltaTime;
  p.age += params.deltaTime;
  p.lifetime -= params.deltaTime;

  // Respawn dead particles at random positions within domain
  if (p.lifetime <= 0.0) {
    let seed = idx * 1000u + u32(p.age * 100.0);
    p.position = vec3<f32>(
      (hash(seed) - 0.5) * params.domainSize.x,
      (hash(seed + 1u) - 0.5) * params.domainSize.y,
      hash(seed + 2u) * params.domainSize.z
    );
    p.velocity = vec3<f32>(0.0);
    p.lifetime = 5.0 + hash(seed + 3u) * 10.0;
    p.age = 0.0;
  }

  // Wrap particles that leave domain
  let half = params.domainSize * 0.5;
  if (abs(p.position.x) > half.x || abs(p.position.y) > half.y) {
    p.lifetime = 0.0; // will respawn next frame
  }

  particles[idx] = p;
}
