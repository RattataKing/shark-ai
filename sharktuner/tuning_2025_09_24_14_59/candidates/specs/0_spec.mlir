module attributes {transform.with_named_sequence} {
  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op attributes {iree_codegen.tuning_spec_entrypoint} {
    transform.yield %arg0 : !transform.any_op
  }
}