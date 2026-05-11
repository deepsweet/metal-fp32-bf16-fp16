import Foundation
import IOKit.ps
import IOKit.pwr_mgt
import Metal

let numElements = 16_777_216
let numIterations = 2048
let numWarmupBatches = 10
let numMeasuredBatches = 30
let numDispatches = 8
let threadGroupSize = 256

func isPluggedIn() -> Bool {
  guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
    let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef]
  else {
    return false
  }

  var hasBattery = false

  for src in list {
    guard
      let desc = IOPSGetPowerSourceDescription(info, src)?.takeUnretainedValue() as? [String: Any]
    else {
      continue
    }

    if let type = desc[kIOPSTypeKey] as? String,
      type == (kIOPSInternalBatteryType as String),
      let isPresent = desc[kIOPSIsPresentKey] as? Bool,
      isPresent {
      hasBattery = true

      if let state = desc[kIOPSPowerSourceStateKey] as? String,
        state == (kIOPSACPowerValue as String) {
        return true
      }
    }
  }

  return !hasBattery
}

guard isPluggedIn() else {
  fatalError(
    "Benchmark must be run with the MacBook plugged in. Running on battery produces misleading results."
  )
}

let metalSource = try String(contentsOfFile: "shaders.metal", encoding: .utf8)

struct Params {
  var numElements: UInt32
  var numIterations: UInt32
}

struct Result {
  let name: String
  let minMS: Double
  let avgMS: Double
  let stdMS: Double
  let gflops: Double
  let gibPerSec: Double
}

func makeRandomBuffer<T>(device: MTLDevice, count: Int, type: T.Type, fill: (Int) -> T) -> MTLBuffer {
  let stride = MemoryLayout<T>.stride
  let options: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
  let buffer = device.makeBuffer(length: count * stride, options: options)!
  let ptr = buffer.contents().bindMemory(to: T.self, capacity: count)
  for i in 0..<count { ptr[i] = fill(i) }
  return buffer
}

// Apple does not provide a BFloat16 type in Swift
func f32toBF16Bits(_ value: Float) -> UInt16 {
  let bits = value.bitPattern
  let lsb = (bits >> 16) & 1
  let roundingBias: UInt32 = 0x7FFF + lsb
  return UInt16(truncatingIfNeeded: (bits + roundingBias) >> 16)
}

func getGPUCount(for device: MTLDevice) -> Int? {
  let matchDict = IOServiceMatching("IOAccelerator")
  var iterator: io_iterator_t = 0
  guard IOServiceGetMatchingServices(kIOMainPortDefault, matchDict, &iterator) == KERN_SUCCESS
  else { return nil }
  defer { IOObjectRelease(iterator) }

  var service = IOIteratorNext(iterator)
  while service != 0 {
    defer { IOObjectRelease(service) }
    if let coreCount = IORegistryEntryCreateCFProperty(
      service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeUnretainedValue() as? Int {
      return coreCount
    }
    service = IOIteratorNext(iterator)
  }

  return nil
}

func runBatch(
  queue: MTLCommandQueue,
  pipeline: MTLComputePipelineState,
  a: MTLBuffer,
  b: MTLBuffer,
  out: MTLBuffer,
  params: MTLBuffer
) -> Double? {
  let threadsPerTG = MTLSize(width: threadGroupSize, height: 1, depth: 1)
  let threadsPerGrid = MTLSize(width: numElements, height: 1, depth: 1)
  let commandBuffer = queue.makeCommandBuffer()!

  for _ in 0..<numDispatches {
    let enc = commandBuffer.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(a, offset: 0, index: 0)
    enc.setBuffer(b, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    enc.setBuffer(params, offset: 0, index: 3)
    enc.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerTG)
    enc.endEncoding()
  }
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  guard commandBuffer.status == .completed,
    commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
  else { return nil }

  return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000.0
}

func computeStats(
  name: String,
  samples: [Double],
  bytesPerElement: Int
) -> Result {
  let minMS = samples.min() ?? -1
  let avgMS = samples.reduce(0, +) / Double(samples.count)
  let variance = samples.reduce(0.0) { $0 + pow($1 - avgMS, 2) } / Double(max(samples.count, 1))
  let stdMS = sqrt(variance)

  let totalFlops = Double(numElements) * Double(numIterations * 4) * Double(numDispatches)
  let totalBytes = Double(numElements * bytesPerElement * 3 * numDispatches)
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
  pipeline: MTLComputePipelineState,
  a: MTLBuffer,
  b: MTLBuffer,
  out: MTLBuffer,
  bytesPerElement: Int
) -> Result {
  // Warmup
  for _ in 0..<numWarmupBatches {
    _ = runBatch(
      queue: queue,
      pipeline: pipeline,
      a: a,
      b: b,
      out: out,
      params: paramsBuffer
    )
  }

  var samples: [Double] = []
  samples.reserveCapacity(numMeasuredBatches)

  for _ in 0..<numMeasuredBatches {
    if let elapsedMs = runBatch(
      queue: queue,
      pipeline: pipeline,
      a: a,
      b: b,
      out: out,
      params: paramsBuffer
    ) {
      samples.append(elapsedMs)
    }
  }

  return computeStats(
    name: name,
    samples: samples,
    bytesPerElement: bytesPerElement
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

let p16 = try device.makeComputePipelineState(
  function: library.makeFunction(name: "bench_f16")!)
let p32 = try device.makeComputePipelineState(
  function: library.makeFunction(name: "bench_f32")!)
let pbf16 = try device.makeComputePipelineState(
  function: library.makeFunction(name: "bench_bf16")!)

// Pre-warm the GPU by running one dummy batch for each pipeline
let dummyParams = device.makeBuffer(
  length: MemoryLayout<Params>.stride, options: .storageModeShared)!

dummyParams.contents().bindMemory(to: Params.self, capacity: 1).pointee = Params(
  numElements: 1,
  numIterations: 1
)

let pipelines: [(String, MTLComputePipelineState)] = [("fp16", p16), ("fp32", p32), ("bf16", pbf16)]

for (_, pipeline) in pipelines {
  let dummyBuffer = device.makeBuffer(length: 2, options: .storageModeShared)!
  _ = runBatch(
    queue: queue,
    pipeline: pipeline,
    a: dummyBuffer,
    b: dummyBuffer,
    out: dummyBuffer,
    params: dummyParams
  )
}

// Shared parameters buffer
let paramsBuffer = device.makeBuffer(
  length: MemoryLayout<Params>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked])!

paramsBuffer.contents().bindMemory(to: Params.self, capacity: 1).pointee = Params(
  numElements: UInt32(numElements),
  numIterations: UInt32(numIterations)
)

// Run benchmarks
var results: [Result] = []

// FP32
let fp32A = makeRandomBuffer(device: device, count: numElements, type: Float.self) { i in
  let sample = Float((i % 1024) + 1) / 1024.0
  return sample
}
let fp32B = makeRandomBuffer(device: device, count: numElements, type: Float.self) { i in
  let sample = Float(((i * 13) % 1024) + 1) / 2048.0
  return sample
}
let fp32Out = device.makeBuffer(
  length: numElements * MemoryLayout<Float>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked]
)!

results.append(
  benchmark(
    name: "FP32",
    pipeline: p32,
    a: fp32A,
    b: fp32B,
    out: fp32Out,
    bytesPerElement: 4
  )
)

// BF16
let bf16A = makeRandomBuffer(device: device, count: numElements, type: UInt16.self) { i in
  let sample = Float((i % 1024) + 1) / 1024.0
  return f32toBF16Bits(sample)
}
let bf16B = makeRandomBuffer(device: device, count: numElements, type: UInt16.self) { i in
  let sample = Float(((i * 13) % 1024) + 1) / 2048.0
  return f32toBF16Bits(sample)
}
let bf16Out = device.makeBuffer(
  length: numElements * MemoryLayout<UInt16>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked]
)!

results.append(
  benchmark(
    name: "BF16",
    pipeline: pbf16,
    a: bf16A,
    b: bf16B,
    out: bf16Out,
    bytesPerElement: 2
  )
)

// FP16
let fp16A = makeRandomBuffer(device: device, count: numElements, type: UInt16.self) { i in
  let sample = Float((i % 1024) + 1) / 1024.0
  return Float16(sample).bitPattern
}
let fp16B = makeRandomBuffer(device: device, count: numElements, type: UInt16.self) { i in
  let sample = Float(((i * 13) % 1024) + 1) / 2048.0
  return Float16(sample).bitPattern
}
let fp16Out = device.makeBuffer(
  length: numElements * MemoryLayout<UInt16>.stride,
  options: [.storageModeShared, .hazardTrackingModeUntracked]
)!

results.append(
  benchmark(
    name: "FP16",
    pipeline: p16,
    a: fp16A,
    b: fp16B,
    out: fp16Out,
    bytesPerElement: 2
  )
)

print()

let gpuCores = getGPUCount(for: device)

if let gpuCores = gpuCores {
  print("device: \(device.name) @ \(gpuCores) GPU cores")
} else {
  print("device: \(device.name)")
}

for result in results {
  print()
  print("type: \(result.name)")
  print("min: \(String(format: "%.3f", result.minMS)) ms")
  print("avg: \(String(format: "%.3f", result.avgMS)) ms")
  print("sd: \(String(format: "%.3f", result.stdMS)) ms")
  print("perf: \(String(format: "%.2f", result.gflops)) GFLOPS")
  print("throughput: \(String(format: "%.2f", result.gibPerSec)) GiB/s")
}
