use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::rc::Rc;
use std::string::String;
use crate::onnx::onnx_node::{AddNeighbourConnection, OnnxNode};
use crate::onnx::temporary_node::{FromRef, TmpOnnxNode};
// use crate::onnx::onnx_node::{OnnxNode, FunctionNode, InputNode, AddOut};
// use crate::onnx::onnx_node::OnnxNode::{Function, Input};
use crate::parser::onnx_model::onnx_proto3::ModelProto;

#[derive(Debug)]
pub struct OnnxGraph{
    root_node: Rc<RefCell<OnnxNode>>,
    secondary_roots: Vec<Rc<OnnxNode>>
}

impl From<ModelProto> for OnnxGraph{
    fn from(model: ModelProto) -> Self {
        let fun_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.node.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
            .collect();
        let init_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.initializer.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
            .collect();
        let input_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.input.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
            .collect();
        let output_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.output.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
            .collect();

        println!("Function nodes");
        for node in fun_nodes.iter(){
            println!("{:?} => {:?}", node.0, node.1.borrow())
        }
        println!("Initalizer nodes");
        for node in init_nodes.iter(){
            println!("{:?} => {:?}", node.0, node.1.borrow())
        }
        println!("Input nodes");
        for node in input_nodes.iter(){
            println!("{:?} => {:?}", node.0, node.1.borrow())
        }
        println!("Output nodes");
        for node in output_nodes.iter(){
            println!("{:?} => {:?}", node.0, node.1.borrow())
        }

        for f_node in fun_nodes.iter(){

        }

        OnnxNode::add_connection(input_nodes[&"Input3".to_string()].clone(), fun_nodes[&"Convolution28".to_string()].clone());
        let root = input_nodes[&"Input3".to_string()].clone();
        OnnxGraph{
            root_node: root,
            secondary_roots: Vec::default()
        }
    }
}

// impl From<ModelProto> for OnnxGraph{
//     fn from(model: ModelProto) -> Self {
//         fn compute_graph(model: &ModelProto, current_node: &mut OnnxNode, outputs_names: HashSet<&String>) {
//             for node in model.graph.node.iter() {
//                 for input in node.input.iter(){
//                     if outputs_names.contains(input){
//                         current_node.add_output(OnnxNode::new(& node.name));
//                     }
//                 }
//             }
//         };
//         let mut root_node = OnnxNode::new(& model.graph.input[0].name);
//         root_node.add_output(OnnxNode::new(& model.graph.node[0].name));
//         compute_graph(&model, &mut *root_node.outputs[0].borrow_mut(), model.graph.node[0].output.iter().collect());
//         OnnxGraph{
//             root_node: root_node,
//             secondary_roots: Vec::default()
//         }
//     }
// }