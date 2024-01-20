use std::any::Any;
use std::cell::{RefCell};
use std::fmt::{Debug, Display, Formatter};
use std::rc::{Rc, Weak};
use crate::onnx::matrix::{MatrixOperationError, MatrixType, TryOperation1, TryOperation1Attributes, TryOperation2Attributes};
use crate::onnx::matrix::MatrixOperationError::{FunctionNotImplementedError, MismatchTypeError, MissingInputError, NotImplementedError, VoidMatrixError};
use crate::parser::onnx_model::onnx_proto3::{AttributeProto, NodeProto, TensorProto, ValueInfoProto};

//Common trait//////////////////////////////////////////////////////////////////////////////
pub trait HaveOut: Debug{
    fn add_out(&mut self, destination_node: Rc<RefCell<dyn HaveIn>>) -> ();

    fn get_outputs(&self) -> &Vec<Rc<RefCell<dyn HaveIn>>>;

    fn as_any(&self) -> &dyn Any;

    fn try_calculate(&mut self) -> Result<&MatrixType, MatrixOperationError>;

    fn try_get_out_data(&mut self) -> Result<&MatrixType, MatrixOperationError>;
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
    expected_dims: Vec<usize>,
    pub data: Option<MatrixType>
}

impl InputNode {
    fn new(name: String, expected_dims: Vec<usize>) -> Self{
        InputNode{
            node_name: name,
            outputs: Vec::default(),
            expected_dims: expected_dims,
            data: None,
        }
    }

    pub fn try_load_data(&mut self, data: MatrixType) -> Result<(), MatrixOperationError>{
        self.data = Some(data.try_reshape(&self.expected_dims, None)?);
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

    fn try_calculate(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        self.try_get_out_data()
    }

    fn try_get_out_data(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        match &self.data {
            Some(d) => Ok(d),
            None => Err(VoidMatrixError)
        }
    }
}

impl Name for InputNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}

impl From<&ValueInfoProto> for InputNode {
    fn from(value_proto: &ValueInfoProto) -> Self {
        InputNode::new(value_proto.name.to_owned(), MatrixType::from(value_proto).get_dims())
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
    pub op2: Option<fn(&MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>,
    pub op3: Option<fn(&MatrixType, &MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>,
    attributes: Vec<AttributeProto>,
}


impl FunctionNode {
    fn new(name: String, op_type: String, inputs_name: Vec<String>, outputs_name: Vec<String>, data: Option<MatrixType>, op1: Option<fn(&MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>, op2: Option<fn(&MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>, op3: Option<fn(&MatrixType, &MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>>, attributes: Vec<AttributeProto>) -> Self{
        FunctionNode{
            node_name: name,
            inputs: Vec::default(),
            outputs: Vec::default(),
            op_type: op_type,
            inputs_name: inputs_name,
            outputs_name: outputs_name,
            data: data,
            op1: op1,
            op2: op2,
            op3: op3,
            attributes: attributes,
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

    // pub fn calculate(&self, m: MatrixType) -> Result<MatrixType, MatrixOperationError>{
    //     self.op1.unwrap()(&m, &self.attributes)
    // }
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

    fn try_calculate(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        let inputs_data = self.inputs.iter().map(|input_n| {
            match input_n.upgrade() {
                Some(i_n) => Ok(i_n.borrow_mut().try_get_out_data()?.to_owned()),
                None => Err(MissingInputError)
            }
        }).collect::<Result<Vec<MatrixType>, MatrixOperationError>>()?;

        if let Some(op1) = self.op1{
            if inputs_data.len() == 1 {
                self.data = Some(op1(&inputs_data[0], &self.attributes)?);
            }
        } else if let Some(op2) = self.op2{
            if inputs_data.len() == 2 {
                self.data = Some(op2(&inputs_data[0], &inputs_data[1], &self.attributes)?);
            }
        } else if let Some(op3) = self.op3{
            if inputs_data.len() == 3 {
                self.data = Some(op3(&inputs_data[0], &inputs_data[1], &inputs_data[2], &self.attributes)?);
            }
        }


        if self.data.is_some(){
            Ok(&self.data.as_ref().unwrap())
        } else {
            Err(MismatchTypeError)
        }
    }

    fn try_get_out_data(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        if self.data.is_some(){
            Ok(self.data.as_ref().unwrap())
        } else {
            self.try_calculate()
        }
    }
}

impl Name for FunctionNode {
    fn get_name(&self) -> &String {
        &self.node_name
    }
}


impl TryFrom<&NodeProto> for FunctionNode {
    type Error = MatrixOperationError;

    fn try_from(node_proto: &NodeProto) -> Result<Self, Self::Error> {
        let mut op1: Option<fn(&MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>> = None;
        let mut op2: Option<fn(&MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>> = None;
        let mut op3: Option<fn(&MatrixType, &MatrixType, &MatrixType, &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>> = None;
        match node_proto.op_type.as_str() {
            "Relu" => op1 = Some(MatrixType::try_relu_attributes),
            "MaxPool" => op1 = Some(MatrixType::try_max_pool_attributes),
            "GlobalAveragePool" => op1 = Some(MatrixType::try_global_average_pool_attributes),
            "Softmax" => op1 = Some(MatrixType::try_softmax_attributes),
            "Add" => op2 = Some(MatrixType::try_add_attributes),
            "MatMul" => op2 = Some(MatrixType::try_mat_mul_attributes),
            "Conv" => {
                if node_proto.input.len() == 2{
                    op2 = Some(MatrixType::try_conv2_attributes);
                } else {
                    op3 = Some(MatrixType::try_conv3_attributes);
                }
            }
            "Reshape" => op2 = Some(MatrixType::try_reshape_attributes),
            "Concat" => op2 = Some(MatrixType::try_concat_attributes),
            "Dropout" => op2 = Some(MatrixType::try_dropout_attributes),
            _ => return Err(FunctionNotImplementedError)
        };
        Ok(FunctionNode::new(
            format!("{}_{}",node_proto.op_type, node_proto.name),
            node_proto.op_type.to_owned(),
            node_proto.input.to_owned(),
            node_proto.output.to_owned(),
            None,
            op1,
            op2,
            op3,
            node_proto.attribute.clone()
        ))
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

    fn try_calculate(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        self.try_get_out_data()
    }

    fn try_get_out_data(&mut self) -> Result<&MatrixType, MatrixOperationError> {
        Ok(&self.data)
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
    expected_dims: Vec<usize>,
    pub data: Option<MatrixType>
}

impl OutputNode {
    fn new(name: String, expected_dims: Vec<usize>) -> Self{
        OutputNode{
            node_name: name,
            inputs: Vec::default(),
            expected_dims: expected_dims,
            data: None,
        }
    }

    pub fn try_compute_all(&mut self) -> Result<&MatrixType, MatrixOperationError>{
        if self.inputs.len() != 1{
            Err(NotImplementedError)
        } else {
            let input_node = self.inputs[0].upgrade();
            match input_node {
                Some(n) => self.data = Some(n.borrow_mut().try_get_out_data()?.to_owned()),
                None => return Err(MissingInputError)
            };
            Ok(self.data.as_ref().unwrap())
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
        OutputNode::new(value_proto.name.to_owned(), MatrixType::from(value_proto).get_dims())
    }
}

impl Display for OutputNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Output node:{}", self.node_name)
    }
}