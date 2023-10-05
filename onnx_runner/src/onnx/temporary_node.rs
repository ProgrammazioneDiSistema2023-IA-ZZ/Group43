use crate::parser::onnx_model::onnx_proto3::{NodeProto, TensorProto, ValueInfoProto};

#[derive(Debug)]
pub struct TmpOnnxNode<'a>{
    node_name: &'a String,
    inputs_names: &'a Vec<String>,
    outputs_names: &'a Vec<String>
}

impl TmpOnnxNode<'_>{
    fn new<'a>(node_name: &'a String, inputs_names: &'a Vec<String>, outputs_names: &'a Vec<String>) -> TmpOnnxNode<'a>{
        TmpOnnxNode{
            node_name: node_name,
            inputs_names: inputs_names,
            outputs_names: outputs_names
        }
    }
}

pub trait FromRef<T>{
    fn from_ref<'a>(node: &'a T) -> TmpOnnxNode<'a>;
}

// impl FromRef<NodeProto> for TmpOnnxNode<'_>{
//     fn from_ref<'a>(node_proto: &'a NodeProto) -> TmpOnnxNode<'a>{
//         TmpOnnxNode::new(
//             &node_proto.name,
//             &node_proto.input,
//             &node_proto.output
//         )
//     }
// }
//
// impl FromRef<ValueInfoProto> for TmpOnnxNode<'_>{
//     fn from_ref<'a>(value_proto: &'a ValueInfoProto) -> TmpOnnxNode<'a>{
//         TmpOnnxNode::new(
//             &value_proto.name,
//             &Vec::default(),
//             &Vec::default()
//         )
//     }
// }
//
// impl FromRef<TensorProto> for TmpOnnxNode<'_>{
//     fn from_ref<'a>(tensor_proto: &'a TensorProto) -> TmpOnnxNode<'a>{
//         TmpOnnxNode::new(
//             &tensor_proto.name,
//             &Vec::default(),
//             &Vec::default()
//         )
//     }
// }

// impl<'a> From<&'a NodeProto> for TmpOnnxNode<'_>{
//     fn from(node_proto: &'a NodeProto) -> TmpOnnxNode<'a>{
//         TmpOnnxNode::new(
//             &node_proto.name,
//             &node_proto.input,
//             &node_proto.output
//         )
//     }
// }

// impl From<&ValueInfoProto> for TmpOnnxNode<'_>{
//     fn from(value_proto: &ValueInfoProto) -> Self {
//         TmpOnnxNode::new(
//             &value_proto.name,
//             &Vec::default(),
//             &Vec::default()
//         )
//     }
// }
//
// impl From<&TensorProto> for TmpOnnxNode<'_>{
//     fn from(tensor_proto: &TensorProto) -> Self {
//         TmpOnnxNode::new(
//             &tensor_proto.name,
//             &Vec::default(),
//             &Vec::default()
//         )
//     }
// }