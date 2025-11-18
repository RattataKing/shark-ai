#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#device_target_hip = #hal.device.target<"hip", [#executable_target_rocm_hsaco_fb]> : !hal.device
module {
  util.global private @__device_0 = #device_target_hip
  hal.executable private @main_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @main_dispatch_0_matmul_8192x640x2560_f32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @main_dispatch_0_matmul_8192x640x2560_f32() {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x2560xf32>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<readonly:tensor<640x2560xf32>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) {iree_gpu.use_rocdl_buffer_instructions} : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x640xf32>>
          %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 2560], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x2560xf32>> -> tensor<8192x2560xf32>
          %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [640, 2560], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<640x2560xf32>> -> tensor<640x2560xf32>
          %5 = tensor.empty() : tensor<8192x640xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<8192x640xf32>) -> tensor<8192x640xf32>
          %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<8192x2560xf32>, tensor<640x2560xf32>) outs(%6 : tensor<8192x640xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %8 = arith.mulf %in, %in_0 : f32
            %9 = arith.addf %out, %8 : f32
            linalg.yield %9 : f32
          } -> tensor<8192x640xf32>
          iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8192, 640], strides = [1, 1] : tensor<8192x640xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x640xf32>>
          return
        }
      }
    }
  }
  util.global private mutable @main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer
  util.initializer {
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
    %c111411200 = arith.constant 111411200 : index
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%c111411200}
    util.global.store %buffer, @main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer
    util.return
  }
  util.func public @main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %0 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer = util.global.load @main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer
    %c0 = arith.constant 0 : index
    %c83886080 = arith.constant 83886080 : index
    %c6553600 = arith.constant 6553600 : index
    %c90439680 = arith.constant 90439680 : index
    %c20971520 = arith.constant 20971520 : index
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@main_dispatch_0::@rocm_hsaco_fb::@main_dispatch_0_matmul_8192x640x2560_f32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@main_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@main_dispatch_0::@rocm_hsaco_fb::@main_dispatch_0_matmul_8192x640x2560_f32) : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
        (%main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer)[%c0, %c83886080], 
        (%main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer)[%c83886080, %c6553600], 
        (%main_dispatch_0_rocm_hsaco_fb_main_dispatch_0_matmul_8192x640x2560_f32_buffer : !hal.buffer)[%c90439680, %c20971520]
      ]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%1) signal(%fence) commands(%cmd) flags("None")
    %c-1_i32 = arith.constant -1 : i32
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
