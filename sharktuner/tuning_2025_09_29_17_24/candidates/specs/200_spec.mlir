module attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence} {
  transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_param {transform.readonly}) {
    transform.annotate %arg0 "compilation_info" = %arg1 : !transform.any_op, !transform.any_param
    transform.yield 
  }
  transform.named_sequence @match_main_dispatch_0_matmul_1024x1024x1024_i8xi8xi32(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<1024x1024xi8>, %arg2: tensor<1024x1024xi8>, %arg3: tensor<1024x1024xi32>):
      %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg2 : tensor<1024x1024xi8>, tensor<1024x1024xi8>) outs(%arg3 : tensor<1024x1024xi32>) {
      ^bb0(%in: i8, %in_0: i8, %out: i32):
        %2 = arith.extsi %in : i8 to i32
        %3 = arith.extsi %in_0 : i8 to i32
        %4 = arith.muli %2, %3 : i32
        %5 = arith.addi %out, %4 : i32
        linalg.yield %5 : i32
      } -> tensor<1024x1024xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    %0 = transform.param.constant #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>, promote_operands = [0, 1], reduction = [0, 0, 128], subgroup_m_count = 4 : i64, subgroup_n_count = 1 : i64, workgroup = [256, 128, 0]}>, translation_info = <pipeline = LLVMGPUVectorDistribute workgroup_size = [64, 4, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>, llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}}>> -> !transform.any_param
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    %updated_root = transform.foreach_match in %arg0 
        @match_main_dispatch_0_matmul_1024x1024x1024_i8xi8xi32 -> @apply_op_config : (!transform.any_op) -> !transform.any_op
    transform.yield %updated_root : !transform.any_op
  }
}