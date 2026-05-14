// -----// IR Dump After InsertSMTConstraintsPass: iree-codegen-insert-smt-constraints //----- //
func.func @main_dispatch_0_matmul_1024x1280x1280_f16xf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 32, {gpu_pipeline_options = #iree_gpu.pipeline_options<no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x1280xf16, #hal.descriptor_type<storage_buffer>>
  %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<1024x1280xf16, #hal.descriptor_type<storage_buffer>> to memref<1024x1280xf16, #amdgpu.address_space<fat_raw_buffer>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1280x1280xf16, #hal.descriptor_type<storage_buffer>>
  %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<1280x1280xf16, #hal.descriptor_type<storage_buffer>> to memref<1280x1280xf16, #amdgpu.address_space<fat_raw_buffer>>
  %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<1024x1280xf32, #hal.descriptor_type<storage_buffer>>
  %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<1024x1280xf32, #hal.descriptor_type<storage_buffer>> to memref<1024x1280xf32, #amdgpu.address_space<fat_raw_buffer>>
  %6 = iree_codegen.load_from_buffer %1 : memref<1024x1280xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<1024x1280xf16>
  %7 = iree_codegen.load_from_buffer %3 : memref<1280x1280xf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<1280x1280xf16>
  %8 = tensor.empty() : tensor<1024x1280xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<1024x1280xf32>) -> tensor<1024x1280xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<1024x1280xf16>, tensor<1280x1280xf16>) outs(%9 : tensor<1024x1280xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>, promote_operands = [0, 1], promotion_types = [#iree_gpu.derived_thread_config, #iree_gpu.derived_thread_config], reduction = [0, 0, 64], subgroup_basis = [[2, 2, 1], [0, 1, 2]], workgroup = [64, 128, 0]}>, root_op = #iree_codegen.root_op<set = 0>} {
  ^bb0(%in: f16, %in_3: f16, %out: f32):
    %11 = arith.extf %in : f16 to f32
    %12 = arith.extf %in_3 : f16 to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<1024x1280xf32>
  %c0_0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c0_1 = arith.constant 0 : index
  %c1280 = arith.constant 1280 : index
  %c1 = arith.constant 1 : index
  %c1280_2 = arith.constant 1280 : index
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
   knobs = {mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<WMMAR4_F32_16x16x16_F16>]>, reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">], subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], [0, 1, 2]], subgroup_size = #iree_codegen.smt.int_knob<"sg_size">, workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0], workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">]}
   dims(%c1024, %c1280, %c1280_2) {
  ^bb0(%arg0: !smt.int, %arg1: !smt.int, %arg2: !smt.int):
    %c1024_3 = smt.int.constant 1024
    %11 = smt.eq %arg0, %c1024_3 : !smt.int
    iree_codegen.smt.assert %11, "dim_0 ({}) == 1024", %arg0 : !smt.bool, !smt.int
    %c1280_4 = smt.int.constant 1280
    %12 = smt.eq %arg1, %c1280_4 : !smt.int
    iree_codegen.smt.assert %12, "dim_1 ({}) == 1280", %arg1 : !smt.bool, !smt.int
    %c1280_5 = smt.int.constant 1280
    %13 = smt.eq %arg2, %c1280_5 : !smt.int
    iree_codegen.smt.assert %13, "dim_2 ({}) == 1280", %arg2 : !smt.bool, !smt.int
    %c32 = smt.int.constant 32
    %c1024_6 = smt.int.constant 1024
    %c65536 = smt.int.constant 65536
    %c512 = smt.int.constant 512
    %14 = iree_codegen.smt.knob "wg_0" : !smt.int
    %15 = iree_codegen.smt.knob "wg_1" : !smt.int
    %16 = iree_codegen.smt.knob "red_2" : !smt.int
    %17 = iree_codegen.smt.knob "sg_m_cnt" : !smt.int
    %18 = iree_codegen.smt.knob "sg_n_cnt" : !smt.int
    %19 = iree_codegen.smt.knob "sg_size" : !smt.int
    %20 = iree_codegen.smt.knob "wg_size_x" : !smt.int
    %21 = iree_codegen.smt.knob "wg_size_y" : !smt.int
    %22 = iree_codegen.smt.knob "wg_size_z" : !smt.int
    %23 = iree_codegen.smt.knob "mma_idx" : !smt.int
    %c0_7 = smt.int.constant 0
    %c0_8 = smt.int.constant 0
    %24 = smt.int.cmp ge %23, %c0_7
    iree_codegen.smt.assert %24, "mma_idx >= 0" : !smt.bool
    %25 = smt.int.cmp le %23, %c0_8
    iree_codegen.smt.assert %25, "mma_idx <= 0" : !smt.bool
    %26 = iree_codegen.smt.lookup %23 [0] -> [16] : !smt.int
    %27 = iree_codegen.smt.lookup %23 [0] -> [16] : !smt.int
    %28 = iree_codegen.smt.lookup %23 [0] -> [16] : !smt.int
    %29 = smt.int.mul %17, %18
    %30 = smt.int.mul %17, %26
    %31 = smt.int.div %14, %30
    %32 = smt.int.mul %18, %27
    %33 = smt.int.div %15, %32
    %34 = smt.int.div %16, %28
    %35 = smt.eq %19, %c32 : !smt.int
    iree_codegen.smt.assert %35, "sg_size == preferred_subgroup_size" : !smt.bool
    %c1_9 = smt.int.constant 1
    %c10 = smt.int.constant 10
    %36 = smt.int.cmp ge %29, %c1_9
    iree_codegen.smt.assert %36, "sg_num >= 1" : !smt.bool
    %37 = smt.int.cmp le %29, %c10
    iree_codegen.smt.assert %37, "sg_num <= 10" : !smt.bool
    %c0_10 = smt.int.constant 0
    %38 = smt.int.mod %arg0, %14
    %39 = smt.eq %38, %c0_10 : !smt.int
    iree_codegen.smt.assert %39, "dim_0 % wg_0 == 0 ({} % {} == 0)", %arg0, %14 : !smt.bool, !smt.int, !smt.int
    %c0_11 = smt.int.constant 0
    %40 = smt.int.mod %arg1, %15
    %41 = smt.eq %40, %c0_11 : !smt.int
    iree_codegen.smt.assert %41, "dim_1 % wg_1 == 0 ({} % {} == 0)", %arg1, %15 : !smt.bool, !smt.int, !smt.int
    %c0_12 = smt.int.constant 0
    %42 = smt.int.mod %arg2, %16
    %43 = smt.eq %42, %c0_12 : !smt.int
    iree_codegen.smt.assert %43, "dim_2 % red_2 == 0 ({} % {} == 0)", %arg2, %16 : !smt.bool, !smt.int, !smt.int
    %44 = smt.int.cmp ge %14, %26
    iree_codegen.smt.assert %44, "wg_0 >= mma_m" : !smt.bool
    %45 = smt.int.cmp le %14, %arg0
    iree_codegen.smt.assert %45, "wg_0 <= dim_0" : !smt.bool
    %46 = smt.int.cmp ge %15, %27
    iree_codegen.smt.assert %46, "wg_1 >= mma_n" : !smt.bool
    %47 = smt.int.cmp le %15, %arg1
    iree_codegen.smt.assert %47, "wg_1 <= dim_1" : !smt.bool
    %48 = smt.int.cmp ge %16, %28
    iree_codegen.smt.assert %48, "red_2 >= mma_k" : !smt.bool
    %49 = smt.int.cmp le %16, %arg2
    iree_codegen.smt.assert %49, "red_2 <= dim_2" : !smt.bool
    %50 = smt.int.cmp le %14, %c512
    iree_codegen.smt.assert %50, "wg_0 <= 512 (max VGPRs)" : !smt.bool
    %51 = smt.int.cmp le %15, %c512
    iree_codegen.smt.assert %51, "wg_1 <= 512 (max VGPRs)" : !smt.bool
    %52 = smt.int.cmp le %16, %c512
    iree_codegen.smt.assert %52, "red_2 <= 512 (max VGPRs)" : !smt.bool
    %c0_13 = smt.int.constant 0
    %53 = smt.int.mod %14, %26
    %54 = smt.eq %53, %c0_13 : !smt.int
    iree_codegen.smt.assert %54, "wg_0 % mma_m == 0 ({} % {} == 0)", %14, %26 : !smt.bool, !smt.int, !smt.int
    %c0_14 = smt.int.constant 0
    %55 = smt.int.mod %15, %27
    %56 = smt.eq %55, %c0_14 : !smt.int
    iree_codegen.smt.assert %56, "wg_1 % mma_n == 0 ({} % {} == 0)", %15, %27 : !smt.bool, !smt.int, !smt.int
    %c0_15 = smt.int.constant 0
    %57 = smt.int.mod %16, %28
    %58 = smt.eq %57, %c0_15 : !smt.int
    iree_codegen.smt.assert %58, "red_2 % mma_k == 0 ({} % {} == 0)", %16, %28 : !smt.bool, !smt.int, !smt.int
    %59 = smt.int.mul %17, %26, %31
    %60 = smt.int.mul %18, %27, %33
    %c0_16 = smt.int.constant 0
    %61 = smt.int.mod %14, %59
    %62 = smt.eq %61, %c0_16 : !smt.int
    iree_codegen.smt.assert %62, "wg_0 % (sg_m_cnt * mma_m * sg_m) == 0 ({} % {} == 0)", %14, %59 : !smt.bool, !smt.int, !smt.int
    %c0_17 = smt.int.constant 0
    %63 = smt.int.mod %15, %60
    %64 = smt.eq %63, %c0_17 : !smt.int
    iree_codegen.smt.assert %64, "wg_1 % (sg_n_cnt * mma_n * sg_n) == 0 ({} % {} == 0)", %15, %60 : !smt.bool, !smt.int, !smt.int
    %65 = smt.int.mul %34, %28
    %66 = smt.eq %16, %65 : !smt.int
    iree_codegen.smt.assert %66, "red_2 == sg_k * mma_k" : !smt.bool
    %c0_18 = smt.int.constant 0
    %67 = smt.int.mod %16, %26
    %68 = smt.eq %67, %c0_18 : !smt.int
    iree_codegen.smt.assert %68, "red_2 % mma_m == 0 ({} % {} == 0)", %16, %26 : !smt.bool, !smt.int, !smt.int
    %c1_19 = smt.int.constant 1
    %c32_20 = smt.int.constant 32
    %69 = smt.int.cmp ge %17, %c1_19
    iree_codegen.smt.assert %69, "sg_m_cnt >= 1" : !smt.bool
    %70 = smt.int.cmp le %17, %c32_20
    iree_codegen.smt.assert %70, "sg_m_cnt <= 32" : !smt.bool
    %71 = smt.int.cmp ge %18, %c1_19
    iree_codegen.smt.assert %71, "sg_n_cnt >= 1" : !smt.bool
    %72 = smt.int.cmp le %18, %c32_20
    iree_codegen.smt.assert %72, "sg_n_cnt <= 32" : !smt.bool
    %73 = smt.int.cmp ge %31, %c1_19
    iree_codegen.smt.assert %73, "sg_m >= 1" : !smt.bool
    %74 = smt.int.cmp le %31, %c32_20
    iree_codegen.smt.assert %74, "sg_m <= 32" : !smt.bool
    %75 = smt.int.cmp ge %33, %c1_19
    iree_codegen.smt.assert %75, "sg_n >= 1" : !smt.bool
    %76 = smt.int.cmp le %33, %c32_20
    iree_codegen.smt.assert %76, "sg_n <= 32" : !smt.bool
    %77 = smt.int.cmp ge %34, %c1_19
    iree_codegen.smt.assert %77, "sg_k >= 1" : !smt.bool
    %78 = smt.int.cmp le %34, %c32_20
    iree_codegen.smt.assert %78, "sg_k <= 32" : !smt.bool
    %79 = smt.int.mul %17, %18
    %80 = smt.eq %79, %29 : !smt.int
    iree_codegen.smt.assert %80, "sg_m_cnt * sg_n_cnt == sg_num" : !smt.bool
    %81 = smt.int.mul %17, %18, %19
    %82 = smt.int.cmp le %81, %c1024_6
    iree_codegen.smt.assert %82, "total_threads <= max_threads" : !smt.bool
    %83 = smt.int.mul %16, %14
    %84 = smt.int.mul %17, %18, %19
    %c0_21 = smt.int.constant 0
    %85 = smt.int.mod %83, %84
    %86 = smt.eq %85, %c0_21 : !smt.int
    iree_codegen.smt.assert %86, "lhs_tile_elements % thread_count == 0 ({} % {} == 0)", %83, %84 : !smt.bool, !smt.int, !smt.int
    %87 = smt.int.mul %16, %15
    %88 = smt.int.mul %17, %18, %19
    %c0_22 = smt.int.constant 0
    %89 = smt.int.mod %87, %88
    %90 = smt.eq %89, %c0_22 : !smt.int
    iree_codegen.smt.assert %90, "rhs_tile_elements % thread_count == 0 ({} % {} == 0)", %87, %88 : !smt.bool, !smt.int, !smt.int
    %c2 = smt.int.constant 2
    %c2_23 = smt.int.constant 2
    %91 = smt.int.mul %c2, %14, %16
    %92 = smt.int.mul %c2_23, %15, %16
    %93 = smt.int.add %91, %92
    %94 = smt.int.cmp le %93, %c65536
    iree_codegen.smt.assert %94, "shared memory must fit in workgroup memory" : !smt.bool
    %95 = smt.eq %20, %81 : !smt.int
    iree_codegen.smt.assert %95, "wg_size_x == total_threads" : !smt.bool
    %96 = smt.int.cmp le %20, %14
    %97 = smt.int.cmp le %20, %15
    %98 = smt.or %96, %97
    iree_codegen.smt.assert %98, "wg_size_x <= wg_m OR wg_size_x <= wg_n" : !smt.bool
    %c1_24 = smt.int.constant 1
    %99 = smt.eq %21, %c1_24 : !smt.int
    iree_codegen.smt.assert %99, "wg_size_y == 1" : !smt.bool
    %c1_25 = smt.int.constant 1
    %100 = smt.eq %22, %c1_25 : !smt.int
    iree_codegen.smt.assert %100, "wg_size_z == 1" : !smt.bool
  }
  iree_codegen.store_to_buffer %10, %5 : tensor<1024x1280xf32> into memref<1024x1280xf32, #amdgpu.address_space<fat_raw_buffer>>
  return
}
