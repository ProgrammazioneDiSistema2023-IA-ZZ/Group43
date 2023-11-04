use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::fmt::{Display, Formatter};
use std::rc::Rc;
use std::string::String;
use crate::onnx::onnx_node::{FunctionNode, InputNode, AddOut, AddIn};
use crate::onnx::temporary_node::{FromRef, TmpOnnxNode};
// use crate::onnx::onnx_node::{OnnxNode, FunctionNode, InputNode, AddOut};
// use crate::onnx::onnx_node::OnnxNode::{Function, Input};
use crate::parser::onnx_model::onnx_proto3::{GraphProto, ModelProto};

#[derive(Debug)]
pub struct OnnxGraph{
    root_node: Rc<RefCell<InputNode>>,
    // secondary_roots: Vec<Rc<OnnxNode>>
}

impl Display for OnnxGraph{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Onnx graph\n{:?}", self.root_node.borrow())
    }
}

pub trait AddEdge{
    fn add_directional_edge(base_node: Rc<RefCell<dyn AddOut>>, destination_node: Rc<RefCell<dyn AddIn>>) -> (){
        base_node.borrow_mut().add_out(destination_node.clone());
        destination_node.borrow_mut().add_in(Rc::downgrade(&base_node));
    }
}

impl AddEdge for OnnxGraph{}

impl From<ModelProto> for OnnxGraph{
    fn from(model: ModelProto) -> Self {
        let root= Rc::new(RefCell::new(InputNode::from(&model.graph.input[0])));
        let node1 = Rc::new(RefCell::new(FunctionNode::from(&model.graph.node[0])));
        // let out = Rc::new(RefCell::new(OutputNode::from(&model.graph.output[0])));
        // Node::add_directional_edge(root.clone(), node1.clone());
        OnnxGraph::add_directional_edge(root.clone(), node1.clone());
        OnnxGraph{
            root_node: root
        }
        // let fun_nodes: Vec<Rc<RefCell<OnnxNode>>> = model.graph.node.iter()
        //     .map(|n| Rc::new(RefCell::new(OnnxNode::from(n))))
        //     .collect();
        // let init_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.initializer.iter()
        //     .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
        //     .collect();
        // let input_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.input.iter()
        //     .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
        //     .collect();
        // let output_nodes: HashMap<&String, Rc<RefCell<OnnxNode>>> = model.graph.output.iter()
        //     .map(|n| (&n.name, Rc::new(RefCell::new(OnnxNode::from(n)))))
        //     .collect();
        //
        // println!("Function nodes");
        // for node in fun_nodes.iter(){
        //     println!("{:?}", node.borrow())
        // }
        // println!("Initalizer nodes");
        // for node in init_nodes.iter(){
        //     println!("{:?}", node.1.borrow())
        // }
        // println!("Input nodes");
        // for node in input_nodes.iter(){
        //     println!("{:?}", node.1.borrow())
        // }
        // println!("Output nodes");
        // for node in output_nodes.iter(){
        //     println!("{:?}", node.1.borrow())
        // }
        //
        // // let all_possible_inputs = input_nodes.iter().chain(fun_nodes.iter()).chain(init_nodes.iter()).collect::<HashMap<&&String, &Rc<RefCell<OnnxNode>>>>();
        // // let all_possible_outputs = fun_nodes.iter().chain(output_nodes.iter()).collect::<HashMap<&&String, &Rc<RefCell<OnnxNode>>>>();
        // //
        // // let a = input_nodes.values().collect::<Vec<&Rc<RefCell<OnnxNode>>>>();
        // // for (i, f_node) in fun_nodes.iter().enumerate(){
        // //     for in_node_name in f_node.input.iter(){
        // //         // let base_node = input_nodes.get(in_node_name)
        // //         //     .or_else(|| fun_nodes.get(in_node_name)
        // //         //         .or_else(|| init_nodes.get(in_node_name)))
        // //         //     .expect(format!("Base node {} not found", in_node_name).as_str());
        // //         // let destination_node = output_nodes.get(&f_node.name)
        // //         //     .or_else(|| fun_nodes.get(&f_node.name))
        // //         //     .expect(format!("Destination node {} not found", f_node.name).as_str());
        // //         let base_node = vec![input_nodes.get(in_node_name)
        // //             .or_else(|| init_nodes.get(in_node_name))
        // //             .expect(format!("Base node {} not found", in_node_name).as_str())];
        // //         //OnnxNode::add_connection(base_node.clone(), destination_node.clone());
        // //     }
        // // }
        //
        // // OnnxNode::add_connection(input_nodes[&"Input3".to_string()].clone(), fun_nodes[&"Convolution28".to_string()].clone());
        // let root = input_nodes[&"Input3".to_string()].clone();
        // let a = root.borrow().to_string();
        // OnnxGraph{
        //     root_node: root,
        //     secondary_roots: Vec::default()
        // }
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