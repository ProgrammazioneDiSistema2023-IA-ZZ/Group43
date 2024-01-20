use std::cell::{RefCell};
use std::collections::{HashMap};
use std::fmt::{Display, Formatter};
use std::rc::{Rc};
use std::string::String;
use crate::onnx::matrix::{MatrixOperationError, MatrixType};
use crate::onnx::onnx_node::{FunctionNode, InputNode, HaveOut, HaveIn, OutputNode, InitNode, Name};
use crate::parser::onnx_model::onnx_proto3::{ModelProto};

#[derive(Debug)]
pub struct OnnxGraph{
    pub root_node: Rc<RefCell<InputNode>>,
    pub secondaries_roots: Vec<Rc<RefCell<FunctionNode>>>,
    pub fun_nodes: Vec<Rc<RefCell<FunctionNode>>>,
    pub init_nodes: Vec<Rc<RefCell<InitNode>>>,
    pub input_nodes: Vec<Rc<RefCell<InputNode>>>,
    pub output_nodes: Vec<Rc<RefCell<OutputNode>>>
}

impl OnnxGraph {
    pub fn try_load_data(&self, data: MatrixType) -> Result<(), MatrixOperationError> {
        self.root_node.borrow_mut().try_load_data(data)
    }
}

impl Display for OnnxGraph{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Onnx graph:\n{:?}\n{:?}\n{}", self.root_node.borrow(), self.secondaries_roots, self.secondaries_roots.len())
    }
}

pub trait AddEdge{
    fn add_directional_edge(base_node: Rc<RefCell<dyn HaveOut>>, destination_node: Rc<RefCell<dyn HaveIn>>) -> (){
        base_node.borrow_mut().add_out(destination_node.clone());
        destination_node.borrow_mut().add_in(Rc::downgrade(&base_node));
    }
}

impl AddEdge for OnnxGraph{}

impl TryFrom<ModelProto> for OnnxGraph{
    type Error = MatrixOperationError;

    fn try_from(model: ModelProto) -> Result<Self, Self::Error> {
        //Initialize nodes
        let fun_nodes: Vec<Rc<RefCell<FunctionNode>>> = model.graph.node.iter()
            .map(|n| FunctionNode::try_from(n))
            .collect::<Result<Vec<FunctionNode>, MatrixOperationError>>()?
            .into_iter()
            .map(|n| Rc::new(RefCell::new(n)))
            .collect();
        let init_nodes: HashMap<&String, Rc<RefCell<InitNode>>> = model.graph.initializer.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(InitNode::from(n)))))
            .collect();
        let input_nodes: HashMap<&String, Rc<RefCell<InputNode>>> = model.graph.input.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(InputNode::from(n)))))
            .collect();
        let output_nodes: HashMap<&String, Rc<RefCell<OutputNode>>> = model.graph.output.iter()
            .map(|n| (&n.name, Rc::new(RefCell::new(OutputNode::from(n)))))
            .collect();

        //Closure that create edges between a node and his inputs
        let create_edges = |inputs_names: Vec<String>, destination_node: Rc<RefCell<dyn HaveIn>>| {
            for input_name in inputs_names.iter() {
                let base_node: Rc<RefCell<dyn HaveOut>> = if let Some(node) = input_nodes.get(input_name) {
                    node.clone()
                } else if let Some(node) = init_nodes.get(input_name) {
                    node.clone()
                } else if let Some(node) = fun_nodes.iter().find(|node| node.borrow().get_outputs_name().contains(input_name)) {
                    node.clone()
                } else {
                    panic!("No base found for connection: {}", input_name)
                };
                OnnxGraph::add_directional_edge(base_node, destination_node.clone());
            }
        };

        //Create edges from nodes to all functional nodes
        for f_node in fun_nodes.iter(){
            let inputs_names = f_node.borrow().get_inputs_name().to_owned();
            create_edges(inputs_names, f_node.clone());
        }

        //Create edges from nodes to all output nodes
        for out_node in output_nodes.iter(){
            let inputs_names = vec![out_node.1.borrow().get_name().to_owned()];
            create_edges(inputs_names, out_node.1.clone());
        }

        //Define root node
        let root_node = match input_nodes.len() {
            1 => input_nodes.iter().nth(0).unwrap().1.clone(),
            _ => panic!("Number of input nodes not supported")
        };

        //Define vec of secondaries root that not start from the input
        let secondaries_roots = fun_nodes.iter().filter_map(|node| {
            if node.borrow().get_inputs().iter().any(|n| {
                match n.upgrade().unwrap().borrow().as_any().downcast_ref::<FunctionNode>() {
                    Some(_) => return true,
                    None => {}
                };
                match n.upgrade().unwrap().borrow().as_any().downcast_ref::<InputNode>() {
                    Some(_) => true,
                    None => false
                }
            })
            {
                None
            }
            else {
                Some(node.clone())
            }
        }).collect();

        Ok(
            OnnxGraph{
                root_node: root_node,
                secondaries_roots: secondaries_roots,
                fun_nodes: fun_nodes,
                init_nodes: init_nodes.into_iter().map(|(_, n)| n).collect(),
                input_nodes: input_nodes.into_iter().map(|(_, n)| n).collect(),
                output_nodes: output_nodes.into_iter().map(|(_, n)| n).collect(),
            }
        )
    }
}