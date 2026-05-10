import Foundation
import Metal

let NUM_ELEMENTS = 16_777_216
let ITERATIONS_PER_KERNEL = 2048
let WARMUP_BATCHES = 10
let MEASURED_BATCHES = 30
let PASSES_PER_COMMAND_BUFFER = 8
let THREAD_GROUP_SIZE = 256

let metalSource = try String(contentsOfFile: "shaders.metal", encoding: .utf8)

struct Params {
  var n: UInt32
  var iters: UInt32
}

struct Result {
  let name: String
  let minMS: Double
  let avgMS: Double
  let stdMS: Double
  let gflops: Double
  let gibPerSec: Double
}

func makePipeline(device: MTLDevice, library: MTLLibrary, name: String) throws
  -> MTLComputePipelineState
{
  guard let fn = library.makeFunction(name: name) else {
    fatalError("Function \(name) not found in library")
  }
  return try device.makeComputePipelineState(function: fn)
}

func makeRandomBuffer<T>(device: MTLDevice, count: Int, type: T.Type, fill: (Int) -> T) -> MTLBuffer
{
  let stride = MemoryLayout<T>.stride
  let options: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
  let buffer = device.makeBuffer(length: count * stride, options: options)!
  let ptr = buffer.contents().bindMemory(to: T.self, capacity: count)
  for i in 0..<count { ptr[i] = fill(i) }
  return buffer
}

// Apple does not provide a BFloat16 type in Swift
func f32toBF16Bits(_ x: Float) -> UInt16 {
  let bits = x.bitPattern
  let lsb = (bits >> 16) & 1
  let roundingBias: UInt32 = 0x7FFF + lsb
  return UInt16(truncatingIfNeeded: (bits + roundingBias) >> 16)
}

func runBatch(
  queue: MTLCommandQueue,
  pipeline: MTLComputePipelineState,
  a: MTLBuffer,
  b: MTLBuffer,
  out: MTLBuffer,
  params: MTLBuffer,
  n: Int,
  passesPerCommandBuffer: Int
) -> Double {
  let threadsPerTG = MTLSize(width: THREAD_GROUP_SIZE, height: 1, depth: 1)
  let threadsPerGrid = MTLSize(width: n, height: 1, depth: 1)
  let cb = queue.makeCommandBuffer()!

  for _ in 0..<passesPerCommandBuffer {
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(a, offset: 0, index: 0)
    enc.setBuffer(b, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    enc.setBuffer(params, offset: 0, index: 3)
    enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerTG)
    enc.endEncoding()
  }
  cb.commit()
  cb.waitUntilCompleted()

  if cb.status == .completed, cb.gpuEndTime > cb.gpuStartTime {
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000.0
  }
  return -1
}

func computeStats(
  name: String,
  samples: [Double],
  n: Int,
  bytesPerElement: Int,
  iters: Int,
  passesPerCommandBuffer: Int
) -> Result {
  let minMS = samples.min() ?? -1
  let avgMS = samples.reduce(0, +) / Double(samples.count)
  let variance = samples.reduce(0.0) { $0 + pow($1 - avgMS, 2) } / Double(max(samples.count, 1))
  let stdMS = sqrt(variance)

  let totalFlops = Double(n) * Double(iters * 4) * Double(passesPerCommandBuffer)
  let totalBytes = Double(n * bytesPerElement * 3 * passesPerCommandBuffer)
  let avgSeconds = avgMS / 1000.0
  let gflops = avgSeconds > 0 ? totalFlops / avgSeconds / 1e9 : -1
  let gibPerSec = avgSeconds > 0 ? totalBytes / avgSeconds / Double(1 << 30) : -1

  return Result(
    name: name,
    minMS: minMS,
    avgMS: avgMS,
    stdMS: stdMS,
    gflops: gflops,
    gibPerSec: gibPerSec
  )
}

func benchmark(
  name: String,
  queue: MTLCommandQueue,
  pipeline: MTLComputePipelineState,
  a: MTLBuffer,
  b: MTLBuffer,
  out: MTLBuffer,
  params: MTLBuffer,
  n: Int,
  bytesPerElement: Int,
  iters: Int,
  warmupBatches: Int,
  measuredBatches: Int,
  passesPerCommandBuffer: Int
) -> Result {
  // Warmup
  for _ in 0..<warmupBatches {
    _ = runBatch(
      queue: queue,
      pipeline: pipeline,
      a: a,
      b: b,
      out: out,
      params: params,
      n: n,
      passesPerCommandBuffer: passesPerCommandBuffer
    )
  }

  var samples: [Double] = []
  samples.reserveCapacity(measuredBatches)

  for _ in 0..<measuredBatches {
    let ms = runBatch(
      queue: queue,
      pipeline: pipeline,
      a: a,
      b: b,
      out: out,
      params: params,
      n: n,
      passesPerCommandBuffer: passesPerCommandBuffer
    )
    if ms > 0 { samples.append(ms) }
  }

  return computeStats(
    name: name,
    samples: samples,
    n: n,
    bytesPerElement: bytesPerElement,
    iters: iters,
    passesPerCommandBuffer: passesPerCommandBuffer
  )
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

let compileOptions = MTLCompileOptions()
compileOptions.languageVersion = .version3_1

if #available(macOS 15.0, *) {
  compileOptions.mathMode = .fast
} else {
  compileOptions.fastMathEnabled = true
}

let library = try device.makeLibrary(source: metalSource, options: compileOptions)

let p16 = try makePipeline(device: device, library: library, name: "bench_f16")
let p32 = try makePipeline(device: device, library: library, name: "bench_f32")
let pbf16 = try makePipeline(device: device, library: library, name: "bench_bf16")

// Pre-warm the GPU by running one dummy batch for each pipeline
let dummyParams = device.makeBuffer(
  length: MemoryLayout<Params>.stride, options: .storageModeShared)!
dummyParams.contents().bindMemory(to: Params.self, capacity: 1).pointee = Params(n: 1, iters: 1)

for (_, pipeline) in [("fp16", p16), ("fp32", p32)] {
  let dummyBuffer = device.makeBuffer(length: 2, options: .storageModeShared)!
  _ = runBatch(
    queue: queue, pipeline: pipeline, a: dummyBuffer, b: dummyBuffer, out: dummyBuffer,
    params: dummyParams, n: 1, passesPerCommandBuffer: 1)
}

let dummyBuffer = device.makeBuffer(length: 2, options: .storageModeShared)!
_ = runBatch(
  queue: queue, pipeline: pbf16, a: dummyBuffer, b: dummyBuffer, out: dummyBuffer,
  params: dummyParams, n: 1, passesPerCommandBuffer: 1)

// Shared parameters buffer
let paramsBuffer = device.makeBuffer(
  length: MemoryLayout<Params>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked])!
paramsBuffer.contents().bindMemory(to: Params.self, capacity: 1).pointee = Params(
  n: UInt32(NUM_ELEMENTS), iters: UInt32(ITERATIONS_PER_KERNEL))

// Run benchmarks
var results: [Result] = []

// FP32
let fp32_a = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: Float.self) { i in
  Float((i % 1024) + 1) / 1024.0
}
let fp32_b = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: Float.self) { i in
  Float(((i * 13) % 1024) + 1) / 2048.0
}
let fp32_out = device.makeBuffer(
  length: NUM_ELEMENTS * MemoryLayout<Float>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked])!

results.append(
  benchmark(
    name: "FP32",
    queue: queue,
    pipeline: p32,
    a: fp32_a,
    b: fp32_b,
    out: fp32_out,
    params: paramsBuffer,
    n: NUM_ELEMENTS,
    bytesPerElement: 4,
    iters: ITERATIONS_PER_KERNEL,
    warmupBatches: WARMUP_BATCHES,
    measuredBatches: MEASURED_BATCHES,
    passesPerCommandBuffer: PASSES_PER_COMMAND_BUFFER
  ))

// BF16
let bf16_a = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: UInt16.self) { i in
  let x = Float((i % 1024) + 1) / 1024.0
  return f32toBF16Bits(x)
}
let bf16_b = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: UInt16.self) { i in
  let x = Float(((i * 13) % 1024) + 1) / 2048.0
  return f32toBF16Bits(x)
}
let bf16_out = device.makeBuffer(
  length: NUM_ELEMENTS * MemoryLayout<UInt16>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked])!

results.append(
  benchmark(
    name: "BF16",
    queue: queue,
    pipeline: pbf16,
    a: bf16_a,
    b: bf16_b,
    out: bf16_out,
    params: paramsBuffer,
    n: NUM_ELEMENTS,
    bytesPerElement: 2,
    iters: ITERATIONS_PER_KERNEL,
    warmupBatches: WARMUP_BATCHES,
    measuredBatches: MEASURED_BATCHES,
    passesPerCommandBuffer: PASSES_PER_COMMAND_BUFFER
  ))

// FP16
let fp16_a = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: UInt16.self) { i in
  let x = Float((i % 1024) + 1) / 1024.0
  return Float16(x).bitPattern
}
let fp16_b = makeRandomBuffer(device: device, count: NUM_ELEMENTS, type: UInt16.self) { i in
  let x = Float(((i * 13) % 1024) + 1) / 2048.0
  return Float16(x).bitPattern
}
let fp16_out = device.makeBuffer(
  length: NUM_ELEMENTS * MemoryLayout<UInt16>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked])!

results.append(
  benchmark(
    name: "FP16",
    queue: queue,
    pipeline: p16,
    a: fp16_a,
    b: fp16_b,
    out: fp16_out,
    params: paramsBuffer,
    n: NUM_ELEMENTS,
    bytesPerElement: 2,
    iters: ITERATIONS_PER_KERNEL,
    warmupBatches: WARMUP_BATCHES,
    measuredBatches: MEASURED_BATCHES,
    passesPerCommandBuffer: PASSES_PER_COMMAND_BUFFER
  ))

print()
print("device: \(device.name)")

for r in results {
  print()
  print("type: \(r.name)")
  print("min: \(String(format: "%.3f", r.minMS)) ms")
  print("avg: \(String(format: "%.3f", r.avgMS)) ms")
  print("sd: \(String(format: "%.3f", r.stdMS)) ms")
  print("perf: \(String(format: "%.2f", r.gflops)) GFLOPS")
  print("throughput: \(String(format: "%.2f", r.gibPerSec)) GiB/s")
}
