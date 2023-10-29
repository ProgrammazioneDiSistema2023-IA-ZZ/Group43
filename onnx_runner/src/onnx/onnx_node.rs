use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt::{Display, Error, Formatter};
use std::io;
use std::rc::{Rc, Weak};
use crate::parser::onnx_model::onnx_proto3::{NodeProto, TensorProto, ValueInfoProto};
// use crate::onnx::onnx_node::OnnxNode::{Function, Input};

// #[derive(Debug)]
// pub enum OnnxNode{
//     Input(InputNode),
//     Output(OutputNode),
//     Function(FunctionNode)
// }
//
// #[derive(Debug)]
// pub struct InputNode{
//     node_name: String,
//     outputs: Vec<Rc<OnnxNode>>
// }
//
// impl InputNode{
//     pub fn new(name: String) -> Self{
//         InputNode{
//             node_name: name,
//             outputs: Vec::default(),
//         }
//     }
// }
//
// impl AddOut for InputNode{
//     fn add_output(&mut self, node: OnnxNode) -> () {
//         self.outputs.push(Rc::new(node));
//     }
// }
//
// #[derive(Debug)]
// pub struct OutputNode{
//     node_name: String,
//     inputs: Vec<Weak<OnnxNode>>
// }
//
// impl OutputNode{
//     pub fn new(name: String) -> Self{
//         OutputNode{
//             node_name: name,
//             inputs: Vec::default(),
//         }
//     }
// }
//
// #[derive(Debug)]
// pub struct FunctionNode{
//     node_name: String,
//     inputs: Vec<Weak<OnnxNode>>,
//     outputs: Vec<Rc<OnnxNode>>
// }
//
// impl FunctionNode{
//     pub fn new(name: String) -> Self{
//         FunctionNode{
//             node_name: name,
//             inputs: Vec::default(),
//             outputs: Vec::default(),
//         }
//     }
// }

// trait AddOut{
//     fn add_output(base: &mut OnnxNode, node: OnnxNode) -> Result<(), Error>{
//         match base {
//             Input(i) | Function(i) => Ok(()),
//             _ => Error
//         }
//     }
// }

#[derive(Debug)]
pub struct OnnxNode{
    node_name: String,
    inputs: Vec<Weak<RefCell<OnnxNode>>>,
    outputs: Vec<Rc<RefCell<OnnxNode>>>,
    inputs_names: Option<HashSet<String>>,
    outputs_names: Option<HashSet<String>>
}

impl OnnxNode{
    fn new(name: & String, inputs_name: Option<HashSet<String>>, outputs_names: Option<HashSet<String>>) -> Self{
        OnnxNode{
            node_name: name.to_owned(),
            inputs: Vec::default(),
            outputs: Vec::default(),
            inputs_names: inputs_name,
            outputs_names: outputs_names
        }
    }

    fn get_name(&self) -> &String{
        &self.node_name
    }

    fn has_inputs(&self) -> bool{
        !self.inputs.is_empty()
    }

    fn has_outputs(&self) -> bool{
        !self.outputs.is_empty()
    }
}

impl Display for OnnxNode{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut out = format!("Node: {}\nInputs({}, len:{}):\n", self.node_name, self.node_name, self.inputs.len());
        self.inputs.iter().for_each(|node| {
            if let Some(n) = node.upgrade(){
                out += format!("{},", n.borrow().node_name).as_str()
            }
        });
        out += format!("\nOutputs({}, len:{}):\n", self.node_name, self.outputs.len()).as_str();
        self.outputs.iter().for_each(|node| out += format!("{},\n", node.borrow()).as_str());
        write!(f, "{}", out)
    }
}

impl From<&NodeProto> for OnnxNode{
    fn from(node_proto: &NodeProto) -> Self {
        OnnxNode::new(
            &format!("{}-{}", node_proto.op_type, node_proto.name),
            Some(node_proto.input.iter().map(|name| name.to_owned()).collect::<HashSet<String>>()),
            Some(node_proto.output.iter().map(|name| name.to_owned()).collect::<HashSet<String>>())
        )
    }
}

impl From<&ValueInfoProto> for OnnxNode{
    fn from(value_proto: &ValueInfoProto) -> Self {
        OnnxNode::new(
            &value_proto.name,
            None,
            None
        )
    }
}

impl From<&TensorProto> for OnnxNode{
    fn from(tensor_proto: &TensorProto) -> Self {
        OnnxNode::new(
            &tensor_proto.name,
            None,
            None
        )
    }
}

pub trait AddNeighbourConnection {
    fn add_connection(base_node:  Rc<RefCell<OnnxNode>>, destination_node:  Rc<RefCell<OnnxNode>>) -> ();
}

impl AddNeighbourConnection for OnnxNode {
    fn add_connection(base_node:  Rc<RefCell<OnnxNode>>, destination_node: Rc<RefCell<OnnxNode>>) -> () {
        base_node.borrow_mut().outputs.push(destination_node.clone());
        destination_node.borrow_mut().inputs.push(Rc::downgrade(&base_node));
    }
}