use std::any::Any;
use std::cell::{RefCell};
use std::fmt::{Debug, Display, Formatter};
use std::rc::{Rc, Weak};
use crate::onnx::matrix::{Data, Matrix, MatrixOperationError, MatrixType, TryOperation1, TryOperation1Attributes, TryOperation2};
use crate::onnx::matrix::MatrixOperationError::MismatchSizeError;
use crate::parser::onnx_model::onnx_proto3::{AttributeProto, NodeProto, TensorProto, ValueInfoProto};

//Common trait//////////////////////////////////////////////////////////////////////////////
pub trait HaveOut: Debug{
    fn add_out(&mut self, destination_node: Rc<RefCell<dyn HaveIn>>) -> ();

    fn get_outputs(&self) -> &Vec<Rc<RefCell<dyn HaveIn>>>;

    fn as_any(&self) -> &dyn Any;
}

pub trait HaveIn: Debug{
    fn add_in(&mut self, base_node: Weak<RefCell<dyn HaveOut>>) -> ();

    fn get_inputs(&self) -> &Vec<Weak<RefCell<dyn HaveOut>>>;

    fn as_any(&self) -> &dyn Any;
}

pub trait Name{
    fn get_name(&self) -> &String;
}

//Input Node////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Default)]
pub struct InputNode {
    node_name: String,
    outputs: Vec<Rc<RefCell<dyn HaveIn>>>,
    pub data: MatrixType
}

impl InputNode {
    fn new(name: String, data: MatrixType) -> Self{
        InputNode{
            node_name: name,
            outputs: Vec::default(),
            data: data
        }
    }

    pub fn try_load_data(&mut self, data: MatrixType) -> Result<(), MatrixOperationError>{
        self.data = data.try_reshape(&self.data.get_dims(), None)?;
        Ok(())
    }
}

impl HaveOut for InputNode {
    fn add_out(&mut self, destination_node: Rc<RefCell<dyn HaveIn>>) -> () {
        self.outputs.push(destination_node);
    }

    fn get_outputs(&self) -> &Vec<Rc<RefCell<dyn HaveIn>>> {
        &self.outputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Name for InputNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}

impl From<&ValueInfoProto> for InputNode {
    fn from(value_proto: &ValueInfoProto) -> Self {
        InputNode::new(value_proto.name.to_owned(), MatrixType::from(value_proto))
    }
}

impl Display for InputNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input node:{}", self.node_name)
    }
}

//Function Node////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Default)]
pub struct FunctionNode{
    node_name: String,
    inputs: Vec<Weak<RefCell<dyn HaveOut>>>,
    outputs: Vec<Rc<RefCell<dyn HaveIn>>>,
    op_type: String,
    inputs_name: Vec<String>,
    outputs_name: Vec<String>,
    data: Option<MatrixType>,
    pub op1: Option<fn(&MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>,
    attributes: Vec<AttributeProto>,
}


impl FunctionNode {
    fn new(name: String, op_type: String, inputs_name: Vec<String>, outputs_name: Vec<String>, data: Option<MatrixType>, op1: Option<fn(&MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>, attributes: Vec<AttributeProto>) -> Self{
        FunctionNode{
            node_name: name,
            inputs: Vec::default(),
            outputs: Vec::default(),
            op_type: op_type,
            inputs_name: inputs_name,
            outputs_name: outputs_name,
            data: data,
            op1: op1,
            attributes: attributes
        }
    }

    pub fn get_inputs_name(&self) -> &Vec<String> {
        &self.inputs_name
    }

    pub fn get_outputs_name(&self) -> &Vec<String> {
        &self.outputs_name
    }

    pub fn get_operation_name(&self) -> &String {
        &self.op_type
    }

    pub fn calculate(&self, m: MatrixType) -> Result<MatrixType, MatrixOperationError>{
        self.op1.unwrap()(&m, &self.attributes)
    }
}

impl HaveIn for FunctionNode {
    fn add_in(&mut self, base_node: Weak<RefCell<dyn HaveOut>>) -> () {
        self.inputs.push(base_node);
    }

    fn get_inputs(&self) -> &Vec<Weak<RefCell<dyn HaveOut>>> {
        &self.inputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl HaveOut for FunctionNode {
    fn add_out(&mut self, destination_node: Rc<RefCell<dyn HaveIn>>) -> () {
        self.outputs.push(destination_node);
    }

    fn get_outputs(&self) -> &Vec<Rc<RefCell<dyn HaveIn>>> {
        &self.outputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Name for FunctionNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}

// Option<fn(&MatrixType) -> Result<MatrixType, MatrixOperationError>>

impl From<&NodeProto> for FunctionNode {
    fn from(node_proto: &NodeProto) -> Self {
        let mut op: Option<fn(&MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>> = None;
        if node_proto.op_type == "MaxPool"{
            op = Some(MatrixType::try_max_pool_attributes);
        }
        FunctionNode::new(
            format!("{}_{}",node_proto.op_type, node_proto.name),
            node_proto.op_type.to_owned(),
            node_proto.input.to_owned(),
            node_proto.output.to_owned(),
            None,
            op,
            node_proto.attribute.clone()
        )
    }
}

impl Display for FunctionNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Function node:{}", self.node_name)
    }
}

//Init Node////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Default)]
pub struct InitNode {
    node_name: String,
    outputs: Vec<Rc<RefCell<dyn HaveIn>>>,
    pub data: MatrixType
}

impl InitNode {
    fn new(name: String, data: MatrixType) -> Self{
        InitNode{
            node_name: name,
            outputs: Vec::default(),
            data: data
        }
    }
}

impl HaveOut for InitNode {
    fn add_out(&mut self, destination_node: Rc<RefCell<dyn HaveIn>>) -> () {
        self.outputs.push(destination_node);
    }

    fn get_outputs(&self) -> &Vec<Rc<RefCell<dyn HaveIn>>> {
        &self.outputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Name for InitNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}

impl From<&TensorProto> for InitNode {
    fn from(tensor_proto: &TensorProto) -> Self {
        InitNode::new(tensor_proto.name.to_owned(), MatrixType::from(tensor_proto))
    }
}

impl Display for InitNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Init node:{}", self.node_name)
    }
}

//Output Node////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Default)]
pub struct OutputNode {
    node_name: String,
    inputs: Vec<Weak<RefCell<dyn HaveOut>>>,
    pub data: MatrixType
}

impl OutputNode {
    fn new(name: String, data: MatrixType) -> Self{
        OutputNode{
            node_name: name,
            inputs: Vec::default(),
            data: data
        }
    }
}

impl HaveIn for OutputNode {
    fn add_in(&mut self, base_node: Weak<RefCell<dyn HaveOut>>) -> () {
        self.inputs.push(base_node);
    }

    fn get_inputs(&self) -> &Vec<Weak<RefCell<dyn HaveOut>>> {
        &self.inputs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Name for OutputNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}

impl From<&ValueInfoProto> for OutputNode {
    fn from(value_proto: &ValueInfoProto) -> Self {
        OutputNode::new(value_proto.name.to_owned(), MatrixType::from(value_proto))
    }
}

impl Display for OutputNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Output node:{}", self.node_name)
    }
}




// #[derive(Debug)]
// pub struct OnnxNode{
//     node_name: String,
//     inputs: Vec<Weak<RefCell<OnnxNode>>>,
//     outputs: Vec<Rc<RefCell<OnnxNode>>>,
//     inputs_names: Option<HashSet<String>>,
//     outputs_names: Option<HashSet<String>>
// }
//
//
//
// impl OnnxNode{
//     fn new(name: & String, inputs_name: Option<HashSet<String>>, outputs_names: Option<HashSet<String>>) -> Self{
//         OnnxNode{
//             node_name: name.to_owned(),
//             inputs: Vec::default(),
//             outputs: Vec::default(),
//             inputs_names: inputs_name,
//             outputs_names: outputs_names
//         }
//     }
//
//     fn get_name(&self) -> &String{
//         &self.node_name
//     }
//
//     fn has_inputs(&self) -> bool{
//         !self.inputs.is_empty()
//     }
//
//     fn has_outputs(&self) -> bool{
//         !self.outputs.is_empty()
//     }
// }
//
// impl Display for OnnxNode{
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut out = format!("Node: {}\nInputs({}, len:{}):\n", self.node_name, self.node_name, self.inputs.len());
//         self.inputs.iter().for_each(|node| {
//             if let Some(n) = node.upgrade(){
//                 out += format!("{},", n.borrow().node_name).as_str()
//             }
//         });
//         out += format!("\nOutputs({}, len:{}):\n", self.node_name, self.outputs.len()).as_str();
//         self.outputs.iter().for_each(|node| out += format!("{},\n", node.borrow()).as_str());
//         write!(f, "{}", out)
//     }
// }
//
// impl From<&NodeProto> for OnnxNode{
//     fn from(node_proto: &NodeProto) -> Self {
//         OnnxNode::new(
//             &format!("{}-{}", node_proto.op_type, node_proto.name),
//             Some(node_proto.input.iter().map(|name| name.to_owned()).collect::<HashSet<String>>()),
//             Some(node_proto.output.iter().map(|name| name.to_owned()).collect::<HashSet<String>>())
//         )
//     }
// }
//
// impl From<&ValueInfoProto> for OnnxNode{
//     fn from(value_proto: &ValueInfoProto) -> Self {
//         OnnxNode::new(
//             &value_proto.name,
//             None,
//             None
//         )
//     }
// }
//
// impl From<&TensorProto> for OnnxNode{
//     fn from(tensor_proto: &TensorProto) -> Self {
//         OnnxNode::new(
//             &tensor_proto.name,
//             None,
//             None
//         )
//     }
// }
//
// pub trait AddNeighbourConnection {
//     fn add_connection(base_node:  Rc<RefCell<OnnxNode>>, destination_node:  Rc<RefCell<OnnxNode>>) -> ();
// }
//
// impl AddNeighbourConnection for OnnxNode {
//     fn add_connection(base_node:  Rc<RefCell<OnnxNode>>, destination_node: Rc<RefCell<OnnxNode>>) -> () {
//         base_node.borrow_mut().outputs.push(destination_node.clone());
//         destination_node.borrow_mut().inputs.push(Rc::downgrade(&base_node));
//     }
// }