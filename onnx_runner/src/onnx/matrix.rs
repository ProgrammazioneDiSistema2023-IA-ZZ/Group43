use std::error::Error;
use std::fmt::{Display, Formatter, write};
use std::ops::Add;
use crate::onnx::matrix::MatrixType::{FloatMatrix, IntMatrix};
use crate::onnx::matrix::OperationError::{DontLastMatrixError, MismatchSizeError, VoidMatrixError, MatrixCompositionError, MismatchTypeError, NotImplementedError};
use crate::parser::onnx_model::onnx_proto3::{TensorProto, ValueInfoProto};
use crate::parser::onnx_model::onnx_proto3::tensor_proto::DataType;
use crate::parser::onnx_model::onnx_proto3::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use crate::parser::onnx_model::onnx_proto3::type_proto::Value;

#[derive(Debug)]
pub enum OperationError{
    VoidMatrixError,
    MismatchSizeError,
    DontLastMatrixError,
    MatrixCompositionError,
    MismatchTypeError,
    NotImplementedError,
}

impl Error for OperationError{}

impl Display for OperationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            VoidMatrixError => write!(f, "The matrix contain no data"),
            MismatchSizeError => write!(f, "The size of the matrices don't match"),
            DontLastMatrixError => write!(f, "The matrix have sub-matrices"),
            MatrixCompositionError => write!(f, "The matrices have wrong compositions"),
            MismatchTypeError => write!(f, "The matrices have wrong types"),
            NotImplementedError => write!(f, "Operation not implemented"),
        }
    }
}

pub trait Load<T>{
    fn load_data(&mut self, data: Vec<T>);
}

pub trait Data<'a, T>{
    type Output;

    fn get_data_or_error(&'a self) ->  Result<Self::Output, OperationError>;
}

pub trait TryOperation1{
    type Output;

    fn try_relu(&self) -> Result<Self::Output, OperationError>;

    fn try_reshape(&self, dim: Vec<usize>) -> Result<Self::Output, OperationError>;
}

pub trait TryOperation2<T>{
    type Output;

    fn try_add(&self, other: T) -> Result<Self::Output, OperationError>;
}

#[derive(Debug, Clone)]
struct Matrix2D<T>{
    rows: usize,
    cols: usize,
    data: Option<Vec<T>>
}

impl<T: Clone + Add<T>> Matrix2D<T> {
    fn new(rows: usize, cols: usize, data: Option<Vec<T>>) -> Self{
        Matrix2D{
            rows,
            cols,
            data
        }
    }
}

impl<T> Load<T> for Matrix2D<T>{
    fn load_data(&mut self, data: Vec<T>) {
        self.data = Some(data)
    }
}

impl<'a, T: 'a> Data<'a, T> for Matrix2D<T> {
    type Output = &'a Vec<T>;

    fn get_data_or_error(&'a self) -> Result<Self::Output, OperationError> {
        match &self.data {
            Some(d) => Ok(d),
            None => Err(VoidMatrixError)
        }
    }
}

impl<T: Copy + Add<Output = T> + Default + PartialOrd> TryOperation1 for Matrix2D<T>{
    type Output = Matrix2D<T>;

    fn try_relu(&self) -> Result<Self::Output, OperationError> {
        let data = self.get_data_or_error()?;
        let zero = T::default();
        let data_out = data.iter().map(|d| {
            if *d > zero{
                *d
            }else{
                zero
            }
        }).collect();
        Ok(Matrix2D::new(
            self.rows,
            self.cols,
            Some(data_out)
        ))
    }

    fn try_reshape(&self, dim: Vec<usize>) -> Result<Self::Output, OperationError> {
        Err(NotImplementedError)
    }
}

impl<T: Copy + Add<Output = T>> TryOperation2<&Matrix2D<T>> for Matrix2D<T>{
    type Output = Matrix2D<T>;

    fn try_add(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, OperationError> {
        let data1 = self.get_data_or_error()?;
        let data2 = other.get_data_or_error()?;
        if self.rows != other.rows || self.cols != other.cols{
            return Err(MismatchSizeError);
        }
        let data_out = data1.iter().zip(data2.iter()).map(|(d1, d2)| *d1 + *d2).collect();
        Ok(Matrix2D::new(
            self.rows,
            self.cols,
            Some(data_out)
        ))
    }
}

#[derive(Debug, Clone)]
pub struct Matrix<T>{
    dims: Vec<usize>,
    matrix2d: Option<Matrix2D<T>>,
    sub_matrices: Option<Vec<Matrix<T>>>
}

impl<T: Copy + Add<Output= T>> Matrix<T> {
    pub fn new(dims: Vec<usize>, data: Option<Vec<T>>) -> Self{
        match dims.len() {
            1 => Matrix{
                    dims: vec![1, dims[0]],
                    matrix2d: Some(Matrix2D::new(1, dims[0], data)),
                    sub_matrices: None
                },
            2 => {
                let rows = dims[0];
                let cols = dims[1];
                Matrix {
                    dims: dims,
                    matrix2d: Some(Matrix2D::new(rows, cols, data)),
                    sub_matrices: None,
                }
            },
            _ => {
                let sub_dims = &dims[1..];
                let sub_data_count = sub_dims.iter().fold(1, |acc, val| acc * val);
                let mut sub_matrices = Vec::new();
                for i in 0..dims[0] {
                    let sub_data = match &data {
                        Some(d) => Some(d[i*sub_data_count..(i+1)*sub_data_count].to_owned()),
                        None => None
                    };
                    sub_matrices.push(Matrix::new(Vec::from(sub_dims), sub_data));
                }
                Matrix{
                    dims: dims,
                    matrix2d: None,
                    sub_matrices: Some(sub_matrices)
                }
            }
        }
    }

    fn new_with_matrix2d(dims: Vec<usize>, matrix2d: Matrix2D<T>) -> Self{
        Matrix{
            dims: dims,
            matrix2d: Some(matrix2d),
            sub_matrices: None
        }
    }

    fn new_with_sub_matrices(dims: Vec<usize>, sub_matrices: Vec<Matrix<T>>) -> Self{
        Matrix{
            dims: dims,
            matrix2d: None,
            sub_matrices: Some(sub_matrices)
        }
    }
}

impl<T: Copy> Load<T> for Matrix<T>{
    fn load_data(&mut self, data: Vec<T>) {
        if let Some(matrix2d) = &mut self.matrix2d {
            matrix2d.load_data(data);
        } else if let Some(sub_matrices) = &mut self.sub_matrices{
            self.matrix2d.iter_mut().enumerate().for_each(|(i, m2d)| {
                let sub_data_count = self.dims[1..].iter().fold(1, |acc, val| acc * val);
                m2d.load_data(data[i * sub_data_count..(i + 1) * sub_data_count].to_owned())
            })
        }
    }
}

impl<'a, T: Copy> Data<'a, T> for Matrix<T> {
    type Output = Vec<T>;

    fn get_data_or_error(&self) -> Result<Self::Output, OperationError> {
        if let Some(m2d) = &self.matrix2d{
            let data = m2d.get_data_or_error()?;
            Ok(data.to_owned())
        }
        else if let Some(sub) = &self.sub_matrices{
            let subs = sub.iter().map(|s| s.get_data_or_error()).collect::<Result<Vec<Vec<T>>, OperationError>>()?;
            let sub_out = subs.into_iter().flatten().collect();
            Ok(sub_out)
        }
        else {
            Err(VoidMatrixError)
        }
    }
}

impl<T: Copy + Add<Output = T> + Default + PartialOrd> TryOperation1 for Matrix<T>{
    type Output = Matrix<T>;

    fn try_relu(&self) -> Result<Self::Output, OperationError> {
        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_relu()?;
            Ok(Matrix::new_with_matrix2d(self.dims.to_owned(), m2d_out))
        }
        else if let Some(sub) = &self.sub_matrices{
            let sub_out = sub.iter().map(|s| s.try_relu()).collect::<Result<Vec<Matrix<T>>, OperationError>>()?;
            Ok(Matrix::new_with_sub_matrices(self.dims.to_owned(), sub_out))
        }
        else {
            Err(VoidMatrixError)
        }
    }

    fn try_reshape(&self, dim: Vec<usize>) -> Result<Self::Output, OperationError> {
        if self.dims.iter().fold(1, |acc, val| acc * val) != dim.iter().fold(1, |acc, val| acc * val){
            return Err(MismatchSizeError);
        }
        let data = self.get_data_or_error()?;
        Ok(Matrix::new(dim, Some(data)))
    }
}

impl<T: Copy + Add<Output= T>> TryOperation2<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;

    fn try_add(&self, other: &Matrix<T>) -> Result<Matrix<T>, OperationError> {
        if let (Some(m2d1), Some(m2d2)) = (&self.matrix2d, &other.matrix2d){
            let m2d_out = m2d1.try_add(m2d2)?;
            Ok(Matrix::new_with_matrix2d(self.dims.to_owned(), m2d_out))
        }
        else if let (Some(sub1), Some(sub2)) = (&self.sub_matrices, &other.sub_matrices){
            let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_add(s2)).collect::<Result<Vec<Matrix<T>>, OperationError>>()?;
            Ok(Matrix::new_with_sub_matrices(self.dims.to_owned(), sub_out))
        }
        else {
            Err(MatrixCompositionError)
        }
    }
}

#[derive(Debug, Clone)]
pub enum MatrixType{
    IntMatrix(Matrix<i64>),
    FloatMatrix(Matrix<f32>)
}

impl From<TensorProto> for MatrixType {
    fn from(tensor_proto: TensorProto) -> Self {
        match tensor_proto.data_type.enum_value() {
            Ok(data_type) => {
                match data_type{
                    DataType::INT64 => IntMatrix(Matrix::new(
                        tensor_proto.dims.iter().map(|d| *d as usize).collect(),
                        Some(tensor_proto.int64_data)
                    )),
                    DataType::FLOAT => FloatMatrix(Matrix::new(
                        tensor_proto.dims.iter().map(|d| *d as usize).collect(),
                        Some(tensor_proto.float_data)
                    )),
                    _ => panic!("Not supported data type")
                }
            },
            Err(_) => panic!("No valid data type")
        }
    }
}

impl From<ValueInfoProto> for MatrixType {
    fn from(value_proto: ValueInfoProto) -> Self {
        let Value::TensorType(tensor) = value_proto.type_.value.as_ref().unwrap();
        let dims = tensor.shape.dim.iter().filter_map(|d| {
            match &d.value {
                Some(value) => {
                    match value {
                        DimValue(v) => Some(*v as usize),
                        DimParam(_) => panic!("Type not implemented")
                    }
                },
                None => None
            }
        }).collect::<Vec<usize>>();
        match tensor.elem_type.enum_value() {
            Ok(data_type) => {
                match data_type{
                    DataType::INT64 => IntMatrix(Matrix::new(
                        dims,
                        None
                    )),
                    DataType::FLOAT => FloatMatrix(Matrix::new(
                        dims,
                        None
                    )),
                    _ => panic!("Not supported data type")
                }
            },
            Err(_) => panic!("No valid data type")
        }
    }
}

impl TryOperation1 for MatrixType{
    type Output = Self;

    fn try_relu(&self) -> Result<Self::Output, OperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_relu()?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_relu()?))
        }
    }

    fn try_reshape(&self, dim: Vec<usize>) -> Result<Self::Output, OperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_reshape(dim)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_reshape(dim)?))
        }
    }
}

impl TryOperation2<&Self> for MatrixType {
    type Output = Self;

    fn try_add(&self, other: &Self) -> Result<Self::Output, OperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_add(other_matrix)?)),
                    FloatMatrix(_) => Err(OperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(OperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_add(other_matrix)?))
                }
            }
        }
    }
}