use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul};
use std::sync::Arc;
use std::thread;
use crate::onnx::matrix::MatrixType::{FloatMatrix, IntMatrix};
use crate::onnx::matrix::MatrixOperationError::{DontLastMatrixError, MismatchSizeError, VoidMatrixError, MatrixCompositionError, MismatchTypeError, NotImplementedError, MissingFieldError, InvalidArgumentError, MissingInputError, ThreadingError, FunctionNotImplementedError };
use crate::parser::onnx_model::onnx_proto3::{AttributeProto, TensorProto, ValueInfoProto};
use crate::parser::onnx_model::onnx_proto3::tensor_proto::DataType;
use crate::parser::onnx_model::onnx_proto3::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use crate::parser::onnx_model::onnx_proto3::type_proto::Value;


#[derive(Debug)]
pub enum MatrixOperationError {
    VoidMatrixError,
    MismatchSizeError,
    DontLastMatrixError,
    MatrixCompositionError,
    MismatchTypeError,
    NotImplementedError,
    MissingFieldError,
    InvalidArgumentError,
    MissingInputError,
    ThreadingError,
    FunctionNotImplementedError,
}

impl Error for MatrixOperationError {}

impl Display for MatrixOperationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            VoidMatrixError => write!(f, "The matrix contain no data"),
            MismatchSizeError => write!(f, "The size of the matrices don't match"),
            DontLastMatrixError => write!(f, "The matrix have sub-matrices"),
            MatrixCompositionError => write!(f, "The matrices have wrong compositions"),
            MismatchTypeError => write!(f, "The matrices have wrong types"),
            NotImplementedError => write!(f, "Operation not implemented"),
            MissingFieldError => write!(f, "Missing field"),
            InvalidArgumentError => write!(f, "Invalid argument"),
            MissingInputError => write!(f, "Missing input"),
            ThreadingError => write!(f, "Threading error"),
            FunctionNotImplementedError => write!(f, "Function not yet implemented error")
        }
    }
}

pub trait Numeric: Copy + Default + Debug + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self> + PartialOrd + Send + Sync + 'static{}

impl Numeric for f32{}
impl Numeric for i64{}

pub trait Load<T>{
    fn load_data(&mut self, data: Vec<T>);
}

pub trait Data<'a, T>{
    type Output;

    fn get_data_or_error(&'a self) ->  Result<Self::Output, MatrixOperationError>;

    fn get_dims(&self) -> Vec<usize>;
}

pub trait TryOperation1{
    type Output;

    fn try_relu(&self) -> Result<Self::Output, MatrixOperationError>;

    fn try_reshape(&self, dim: &Vec<usize>, allow_zero: Option<&usize>) -> Result<Self::Output, MatrixOperationError>;

    fn try_broadcast(&self, dim: Arc<Vec<usize>>) -> Result<Self::Output, MatrixOperationError>;

    fn try_max_pool(&self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, ceil_mode: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>, storage_order: Arc<Option<usize>>) -> Result<Self::Output, MatrixOperationError>;

    fn try_global_max_pool(&self) -> Result<Self::Output, MatrixOperationError>;
}

pub trait TryOperation1FloatOnly{
    type Output;

    fn try_softmax(&self, axis: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError>;

    fn try_global_average_pool(&self) -> Result<Self::Output, MatrixOperationError>;

}

pub trait TryOperation2<T>{
    type Output;

    fn try_add(&self, other: T) -> Result<Self::Output, MatrixOperationError>;

    fn try_mat_mul(&self, other: T) -> Result<Self::Output, MatrixOperationError>;

    fn try_conv2(&self, kernel: T, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError>;

    fn try_conv3(&self, kernel: T, bias: T, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError>;

    fn try_concat(&self, other: T, axis: Arc<i64>) -> Result<Self::Output, MatrixOperationError>;
}

pub trait TryOperation2FloatOnly<T>{
    type Output;

    fn try_dropout(&self, ratio: T, seed: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError>;
}

pub trait TryOperation1Attributes{
    fn try_relu_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>;

    fn try_max_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>;

    fn try_global_max_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>;

    fn try_global_average_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>;

    fn try_softmax_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError>;
}

pub trait TryOperation2Attributes{
    type Output;

    fn try_add_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_mat_mul_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_conv2_attributes(&self, kernel: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_conv3_attributes(&self, kernel: &Self, bias: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_reshape_attributes(&self, dim: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_dropout_attributes(&self, ratio: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;

    fn try_concat_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError>;
}

#[derive(Debug, Clone, Default)]
struct Matrix2D<T>{
    rows: usize,
    cols: usize,
    data: Option<Vec<T>>
}

impl<T: Numeric> Matrix2D<T> {
    fn new(rows: usize, cols: usize, data: Option<Vec<T>>) -> Self{
        Matrix2D{
            rows,
            cols,
            data
        }
    }

    fn pad(&self, pads: &Vec<usize>, pad_value: Option<T>) -> Result<Self, MatrixOperationError>{
        let new_rows = self.rows + pads[0] + pads[1];
        let new_cols = self.cols + pads[2] + pads[3];
        let val = match pad_value {
            Some(v) => v,
            None => T::default()
        };
        let old_data = self.get_data_or_error()?;
        let mut data_out = Vec::new();
        let mut id = 0;
        for i in 0..new_rows{
            let is_pad = i < pads[0] || i > new_rows - pads[1] - 1;
            for j in 0..new_cols{
                if is_pad{
                    data_out.push(val);
                    continue;
                }
                if j < pads[2] || j > new_cols - pads[3] - 1{
                    data_out.push(val);
                }else {
                    data_out.push(old_data[id]);
                    id += 1;
                }
            }
        }
        Ok(
            Matrix2D::new(
                new_rows,
                new_cols,
                Some(data_out)
            )
        )
    }
}

impl<T> Load<T> for Matrix2D<T>{
    fn load_data(&mut self, data: Vec<T>) {
        self.data = Some(data)
    }
}

impl<'a, T: 'a> Data<'a, T> for Matrix2D<T> {
    type Output = &'a Vec<T>;

    fn get_data_or_error(&'a self) -> Result<Self::Output, MatrixOperationError> {
        match &self.data {
            Some(d) => Ok(d),
            None => Err(VoidMatrixError)
        }
    }

    fn get_dims(&self) -> Vec<usize> {
        vec![self.rows, self.cols]
    }
}

impl<T: Numeric> TryOperation1 for Matrix2D<T>{
    type Output = Matrix2D<T>;

    fn try_relu(&self) -> Result<Self::Output, MatrixOperationError> {
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

    fn try_reshape(&self, dim: &Vec<usize>, allow_zero: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        Err(NotImplementedError)
    }

    fn try_broadcast(&self, dim: Arc<Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        let mut data = self.get_data_or_error()?.clone();
        if self.cols != dim[1] {
            if self.cols == 1 {
                data = data.iter().map(|d| vec![*d; dim[1]]).flatten().collect();
            } else {
                return Err(MismatchSizeError);
            }
        }

        if self.rows != dim[0]{
            if self.rows == 1{
                data = (0..dim[0]).map(|_| data.clone()).flatten().collect();
            } else {
                return Err(MismatchSizeError);
            }
        }
        Ok(Matrix2D::new(dim[0], dim[1], Some(data)))
    }

    fn try_max_pool(&self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, ceil_mode: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>, storage_order: Arc<Option<usize>>) -> Result<Self::Output, MatrixOperationError> {
        let _strides = match strides.as_ref() {
            Some(s) => s.to_owned(),
            None => vec![1; kernel_shape.len()]
        };
        let data = self.get_data_or_error()?;
        let mut data_out = Vec::new();
        for i_outer in (0..self.rows-kernel_shape[0]+1).step_by(_strides[0]) {
            for j_outer in (0..self.cols-kernel_shape[1]+1).step_by(_strides[1]) {
                let mut max = data[i_outer*self.cols + j_outer];
                for i in i_outer..(i_outer+kernel_shape[0]) {
                    for j in j_outer..(j_outer+kernel_shape[1]) {
                        if data[i*self.cols + j] > max{
                            max = data[i*self.cols + j];
                        }
                    }
                }
                data_out.push(max);
            }
        }
        Ok(Matrix2D::new(
            (self.rows - kernel_shape[0]) / _strides[0] + 1,
            (self.cols - kernel_shape[1]) / _strides[1] + 1,
            Some(data_out)
        ))
    }

    fn try_global_max_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        let kernel_shape = Arc::new(self.get_dims());
        let strides = Arc::new(None);
        let auto_pad = Arc::new(None);
        let pads = Arc::new(None);
        let ceil_mode = Arc::new(None);
        let dilations = Arc::new(None);
        let storage_order = Arc::new(None);
        self.try_max_pool(kernel_shape, strides, auto_pad, pads, ceil_mode, dilations, storage_order)
    }
}

impl TryOperation1FloatOnly for Matrix2D<f32>{
    type Output = Matrix2D<f32>;

    fn try_softmax(&self, axis: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError> {
        Err(NotImplementedError)
    }


    fn try_global_average_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        let data = self.get_data_or_error()?;
        let data_out = vec![data.iter().fold(f32::default(), |acc, val| acc + *val) / (data.len() as f32)];
        Ok(Matrix2D::new(
            1,
            1,
            Some(data_out)
        ))
    }
}

impl<T: Numeric> TryOperation2<&Matrix2D<T>> for Matrix2D<T>{
    type Output = Matrix2D<T>;

    fn try_add(&self, other: &Matrix2D<T>) -> Result<Matrix2D<T>, MatrixOperationError> {
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

    fn try_mat_mul(&self, other: &Matrix2D<T>) -> Result<Self::Output, MatrixOperationError> {
        if self.cols != other.rows{
            return Err(MismatchSizeError);
        }
        let data1 = self.get_data_or_error()?;
        let data2 = other.get_data_or_error()?;
        let mut data_out = Vec::new();
        for n in 0..self.rows{
            let row_cum_id = n * self.cols;
            for m in 0..other.cols{
                let mut sum = T::default();
                for k in 0..self.cols{
                    sum = sum + data1[row_cum_id + k] * data2[k * other.cols + m];
                }
                //out[n,m] = sum
                data_out.push(sum);
            }
        }
        Ok(Matrix2D::new(
            self.rows,
            other.cols,
            Some(data_out)
        ))
    }

    fn try_conv2(&self, kernel: &Matrix2D<T>, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        let _strides = match strides.as_ref() {
            Some(s) => s,
            None => return Err(MissingFieldError)
        };
        let _pads = match pads.as_ref() {
            Some(p) => p,
            None => return Err(MissingFieldError)
        };
        let m2d;
        let mut m2d_ref = self;
        if _pads.iter().any(|p| *p != 0){
            m2d = self.pad(&_pads, None)?;
            m2d_ref = &m2d;
        }
        let data = m2d_ref.get_data_or_error()?;
        let kernel_data = kernel.get_data_or_error()?;
        let mut data_out = Vec::new();
        for i_outer in (0..m2d_ref.rows-kernel.rows+1).step_by(_strides[0]) {
            for j_outer in (0..m2d_ref.cols-kernel.cols+1).step_by(_strides[1]) {
                let mut sum = T::default();
                for i in 0..kernel.rows {
                    for j in 0..kernel.cols {
                        sum = sum + data[(i_outer + i)*m2d_ref.cols + j_outer + j] * kernel_data[i*kernel.cols + j]
                    }
                }
                data_out.push(sum);
            }
        }
        Ok(Matrix2D::new(
            (self.rows - kernel.rows + _pads[0] + _pads[1]) / _strides[0] + 1,
            (self.cols - kernel.cols + _pads[2] + _pads[3]) / _strides[1] + 1,
            Some(data_out)
        ))
    }

    fn try_conv3(&self, kernel: &Matrix2D<T>, bias: &Matrix2D<T>, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        Err(NotImplementedError)
    }

    fn try_concat(&self, other: &Matrix2D<T>, axis: Arc<i64>) -> Result<Self::Output, MatrixOperationError> {
        let data1 = self.get_data_or_error()?;
        let data2 = other.get_data_or_error()?;
        let mut data_out = Vec::new();
        let mut rows = self.rows;
        let mut cols = self.cols;
        match axis.as_ref() {
            0 => {
                if self.cols != other.cols{
                    return Err(MismatchSizeError);
                }
                data_out = data1.iter().chain(data2.iter()).map(|d| *d).collect();
                rows = self.rows + other.rows;
            },
            1 => {
                if self.rows != other.rows{
                    return Err(MismatchSizeError);
                }
                for r in 0..self.rows{
                    let mut r_id = r * self.cols;
                    for c in 0..self.cols{
                        data_out.push(data1[r_id + c]);
                    }
                    r_id = r * other.cols;
                    for c in 0..other.cols{
                        data_out.push(data2[r_id + c]);
                    }
                }
                cols = self.cols + other.cols;
            },
            _ => return Err(InvalidArgumentError)
        };
        Ok(Matrix2D::new(
            rows,
            cols,
            Some(data_out)
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Matrix<T>{
    dims: Vec<usize>,
    matrix2d: Option<Matrix2D<T>>,
    sub_matrices: Option<Vec<Arc<Matrix<T>>>>
}

impl<T: Numeric> Matrix<T> {
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
                    sub_matrices.push(Arc::new(Matrix::new(Vec::from(sub_dims), sub_data)));
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

    fn new_with_sub_matrices(dims: Vec<usize>, sub_matrices: Vec<Arc<Matrix<T>>>) -> Self{
        Matrix{
            dims: dims,
            matrix2d: None,
            sub_matrices: Some(sub_matrices)
        }
    }

    fn get_dim_for_broadcast(dim1: &Vec<usize>, dim2: &Vec<usize>) -> Result<Vec<usize>, MatrixOperationError>{
        let dim_same;
        let dim_enlarged;
        if dim1.len() > dim2.len() {
            dim_same = dim1;
            dim_enlarged = Self::enlarge_dim(dim2, dim1.len());
        } else{
            dim_same = dim2;
            dim_enlarged = Self::enlarge_dim(dim1, dim2.len());
        }
        dim_same.iter().zip(dim_enlarged.into_iter()).map(|(d1, d2)| {
            if *d1 == d2{
                Ok(*d1)
            } else if *d1 == 1 || d2 == 1 {
                Ok(*d1 * d2)
            }else {
                Err(MismatchSizeError)
            }
        }).collect::<Result<Vec<usize>, MatrixOperationError>>()
    }

    fn enlarge_dim(dim: &Vec<usize>, len: usize) -> Vec<usize>{
        let mut v = Vec::new();
        let diff = len - dim.len();
        for i in 0..len {
            if i < diff{
                v.push(1);
            }else {
                v.push(dim[i - diff]);
            }
        }
        v
    }
}

impl<T: Numeric> Load<T> for Matrix<T>{
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

impl<'a, T: Numeric> Data<'a, T> for Matrix<T> {
    type Output = Vec<T>;

    fn get_data_or_error(&self) -> Result<Self::Output, MatrixOperationError> {
        if let Some(m2d) = &self.matrix2d{
            let data = m2d.get_data_or_error()?;
            Ok(data.to_owned())
        }
        else if let Some(sub) = &self.sub_matrices{
            let subs = sub.iter().map(|s| s.get_data_or_error()).collect::<Result<Vec<Vec<T>>, MatrixOperationError>>()?;
            let sub_out = subs.into_iter().flatten().collect();
            Ok(sub_out)
        }
        else {
            Err(VoidMatrixError)
        }
    }

    fn get_dims(&self) -> Vec<usize> {
        self.dims.to_owned()
    }
}

impl<T: Numeric> TryOperation1 for Matrix<T>{
    type Output = Arc<Matrix<T>>;

    fn try_relu(&self) -> Result<Self::Output, MatrixOperationError> {
        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_relu()?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let Some(sub) = &self.sub_matrices{
            let mut sub_out = Vec::new();
            if sub.len() == 1{
                sub_out.push(sub[0].try_relu()?)
            } else {
                let mut handles = Vec::new();
                for s in sub.iter(){
                    let s_new = s.clone();
                    handles.push(thread::spawn(move || s_new.try_relu()));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(VoidMatrixError)
        }
    }

    fn try_reshape(&self, dim: &Vec<usize>, allow_zero: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        if let Some(a) = allow_zero{
            if *a != 0 {
                return Err(NotImplementedError);
            }
        }

        if self.dims.iter().fold(1, |acc, val| acc * val) != dim.iter().fold(1, |acc, val| acc * val){
            return Err(MismatchSizeError);
        }
        let data = self.get_data_or_error()?;
        Ok(Arc::new(Matrix::new(dim.to_owned(), Some(data))))
    }

    fn try_broadcast(&self, dim: Arc<Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        let this_dim = Self::enlarge_dim(&self.dims, dim.len());
        let out = self.try_reshape(&this_dim, None)?;
        if let Some(m2d) = &out.matrix2d{
            let m2d_out = m2d.try_broadcast(dim)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let Some(subs) = &out.sub_matrices {
            let mut sub_out: Vec<Arc<Matrix<T>>>;
            if out.dims[0] == 1{
                if out.dims[0] != dim[0]{
                    sub_out = (0..dim[0]).map(|_| subs[0].clone()).collect();
                }else {
                    sub_out = subs.clone();
                }
            }else {
                if out.dims[0] != dim[0]{
                    return Err(MismatchSizeError);
                }else {
                    sub_out = subs.clone();
                }
            }

            if sub_out.len() == 1{
                sub_out = vec![sub_out[0].try_broadcast(Arc::new(dim[1..].to_vec()))?];
            } else {
                let mut handles = Vec::new();
                for s in sub_out.iter(){
                    let s_new = s.clone();
                    let dim_new =Arc::new(dim[1..].to_vec());
                    handles.push(thread::spawn(move || s_new.try_broadcast(dim_new)));
                }
                sub_out = Vec::new();
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            // sub_out = sub_out.iter().map(|s| s.try_broadcast(&dim[1..].to_vec())).collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_max_pool(&self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, ceil_mode: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>, storage_order: Arc<Option<usize>>) -> Result<Self::Output, MatrixOperationError> {
        if kernel_shape.len() != 2{
            return Err(NotImplementedError);
        }
        if let Some(a) = auto_pad.as_ref() {
            if a != "NOTSET"{
                return Err(NotImplementedError);
            }
        }
        if let Some(p) = pads.as_ref() {
            if p.iter().any(|pad| *pad != 0){
                return Err(NotImplementedError);
            }
        }
        if let Some(c) = ceil_mode.as_ref() {
            if *c != 0 {
                return Err(NotImplementedError);
            }
        }
        if let Some(d) = dilations.as_ref() {
            if d.iter().any(|dil| *dil != 1){
                return Err(NotImplementedError);
            }
        }
        if let Some(s) = storage_order.as_ref() {
            if *s != 0 {
                return Err(NotImplementedError);
            }
        }

        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_max_pool(kernel_shape, strides, auto_pad, pads, ceil_mode, dilations, storage_order)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let Some(sub) = &self.sub_matrices{
            let mut sub_out = Vec::new();
            if sub.len() == 1{
                sub_out.push(sub[0].try_max_pool(kernel_shape, strides, auto_pad, pads, ceil_mode, dilations, storage_order)?)
            } else {
                let mut handles = Vec::new();
                for s in sub.iter(){
                    let s_new = s.clone();
                    let kernel_shape_new = kernel_shape.clone();
                    let strides_new = strides.clone();
                    let auto_pads_new = auto_pad.clone();
                    let pads_new = pads.clone();
                    let ceil_mode_new = ceil_mode.clone();
                    let dilations_new = dilations.clone();
                    let storage_order_new = storage_order.clone();
                    handles.push(
                        thread::spawn(move || s_new.try_max_pool(kernel_shape_new, strides_new, auto_pads_new, pads_new, ceil_mode_new, dilations_new, storage_order_new)));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            // let sub_out = sub.iter().map(|s| s.try_max_pool(kernel_shape, strides, auto_pad, pads, None, None, None)).collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(VoidMatrixError)
        }
    }

    fn try_global_max_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_global_max_pool()?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let Some(sub) = &self.sub_matrices{
            let mut sub_out = Vec::new();
            if sub.len() == 1{
                sub_out.push(sub[0].try_global_max_pool()?)
            } else {
                let mut handles = Vec::new();
                for s in sub.iter(){
                    let s_new = s.clone();
                    handles.push(
                        thread::spawn(move || s_new.try_global_max_pool()));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(VoidMatrixError)
        }
    }
}

impl TryOperation1FloatOnly for Arc<Matrix<f32>> {
    type Output = Arc<Matrix<f32>>;

    fn try_softmax(&self, axis: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError> {
        if let Some(a) = axis.as_ref() {
            if *a != 1 {
                return Err(NotImplementedError);
            }
        }
        if self.dims.len() > 2 && self.dims[0] != 1{
            return Err(NotImplementedError);
        }
        let data = self.get_data_or_error()?;
        let sum = data.iter().fold(f32::default(), |acc, val| acc + val.exp());
        let data_out = data.iter().map(|d| d.exp()/sum).collect();
        Ok(Arc::new(Matrix::new(self.get_dims(), Some(data_out))))
    }

    fn try_global_average_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_global_average_pool()?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let Some(sub) = &self.sub_matrices{
            let mut sub_out = Vec::new();
            if sub.len() == 1{
                sub_out.push(sub[0].try_global_average_pool()?)
            } else {
                let mut handles = Vec::new();
                for s in sub.iter(){
                    let s_new = s.clone();
                    handles.push(
                        thread::spawn(move || s_new.try_global_average_pool()));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(VoidMatrixError)
        }
    }
}

impl<T: Numeric> TryOperation2<Arc<Matrix<T>>> for Arc<Matrix<T>>{
    type Output = Arc<Matrix<T>>;

    fn try_add(&self, other: Arc<Matrix<T>>) -> Result<Self::Output, MatrixOperationError> {
        fn try_broadcast_either<T: Numeric>(this: Arc<Matrix<T>> , other: Arc<Matrix<T>>) -> Result<(Arc<Matrix<T>>, Arc<Matrix<T>>), MatrixOperationError>{
            let broadcast_dim = Arc::new(Matrix::<T>::get_dim_for_broadcast(&this.dims, &other.dims)?);
            let m1 = this.try_broadcast(broadcast_dim.clone())?;
            let m2 = other.try_broadcast(broadcast_dim.clone())?;
            Ok((m1, m2))
        }
        let mut m1 = self.clone();
        let mut m2 = other.clone();
        if self.dims.len() == other.dims.len(){
            if self.dims.iter().zip(other.dims.iter()).any(|(d1, d2)| d1 != d2){
                (m1, m2) = try_broadcast_either(self.clone(), other.clone())?;
            }
        }else{
            (m1, m2) = try_broadcast_either(self.clone(), other.clone())?;
        }

        if let (Some(m2d1), Some(m2d2)) = (&m1.matrix2d, &m2.matrix2d){
            let m2d_out = m2d1.try_add(m2d2)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let (Some(sub1), Some(sub2)) = (&m1.sub_matrices, &m2.sub_matrices){
            let mut sub_out = Vec::new();
            if sub1.len() == 1 && sub2.len() == 1{
                sub_out.push(sub1[0].try_add(sub2[0].clone())?)
            } else {
                let mut handles = Vec::new();
                for (s1, s2) in sub1.iter().zip(sub2.iter()){
                    let s1_new = s1.clone();
                    let s2_new = s2.clone();
                    handles.push(thread::spawn(move || s1_new.try_add(s2_new)));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            // let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_add(s2)).collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_mat_mul(&self, other: Arc<Matrix<T>>) -> Result<Self::Output, MatrixOperationError> {
        let m1 = self.clone();
        let m2 = other;

        if let (Some(m2d1), Some(m2d2)) = (&m1.matrix2d, &m2.matrix2d){
            let m2d_out = m2d1.try_mat_mul(m2d2)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let (Some(sub1), Some(sub2)) = (&m1.sub_matrices, &m2.sub_matrices){
            let mut sub_out = Vec::new();
            if sub1.len() == 1 && sub2.len() == 1{
                sub_out.push(sub1[0].try_mat_mul(sub2[0].clone())?)
            } else {
                let mut handles = Vec::new();
                for (s1, s2) in sub1.iter().zip(sub2.iter()){
                    let s1_new = s1.clone();
                    let s2_new = s2.clone();
                    handles.push(thread::spawn(move || s1_new.try_mat_mul(s2_new)));
                }
                for h in handles{
                    match h.join(){
                        Ok(res) => sub_out.push(res?),
                        Err(_) => return Err(ThreadingError)
                    }
                };
            }

            // let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_mat_mul(s2)).collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_conv2(&self, kernel: Arc<Matrix<T>>, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        fn get_pad_couple(dim: usize, kernel_dim: usize, stride_dim: usize, is_upper: bool) -> Vec<usize>{
            let out_dim = (dim + stride_dim - 1) / stride_dim;
            let total_pads = kernel_dim + stride_dim * (out_dim - 1) - dim;
            let small_pad = total_pads / 2;
            let big_pad = if total_pads%2 == 0 { small_pad } else { small_pad+1 };
            if is_upper{
                vec![small_pad, big_pad]
            } else {
                vec![big_pad, small_pad]
            }
        }

        if let Some(g) = group.as_ref(){
            if *g != 1 {
                return Err(NotImplementedError);
            }
        }
        if let Some(d) = dilations.as_ref(){
            if d.iter().any(|dil| *dil != 1){
                return Err(NotImplementedError);
            }
        }

        let _strides: Arc<Option<Vec<usize>>> = match strides.as_ref() {
            Some(s) => Arc::new(Some(s.to_owned())),
            None => Arc::new(Some(vec![1; kernel_shape.len()]))
        };
        let _pads: Arc<Option<Vec<usize>>> = match auto_pad.as_ref() {
            Some(a_p) => {
                match a_p.as_str() {
                    "NOTSET" => {
                        if let Some(p) = pads.as_ref(){
                            Arc::new(Some(p.into_iter().map(|val| *val).collect()))
                        } else {
                            Arc::new(Some(vec![0; 4]))
                        }
                    }
                    "SAME_UPPER" => {
                        let _s = _strides.as_ref().clone().unwrap();
                        let mut row_pad = get_pad_couple(self.dims[self.dims.len()-2], kernel.dims[kernel.dims.len()-2], _s[0], true);
                        let mut col_pad = get_pad_couple(self.dims[self.dims.len()-1], kernel.dims[kernel.dims.len()-1], _s[1], true);
                        row_pad.append(&mut col_pad);
                        Arc::new(Some(row_pad))
                    }
                    "SAME_LOWER" => {
                        let _s = _strides.as_ref().clone().unwrap();
                        let mut row_pad = get_pad_couple(self.dims[self.dims.len()-2], kernel.dims[kernel.dims.len()-2], _s[0], false);
                        let mut col_pad = get_pad_couple(self.dims[self.dims.len()-1], kernel.dims[kernel.dims.len()-1], _s[1], false);
                        row_pad.append(&mut col_pad);
                        Arc::new(Some(row_pad))
                    }
                    _ => return Err(NotImplementedError)
                }
            }
            None => {
                if let Some(p) = pads.as_ref(){
                    Arc::new(Some(p.into_iter().map(|val| *val).collect()))
                } else {
                    Arc::new(Some(vec![0; 4]))
                }
            }
        };

        if let (Some(m2d1), Some(m2d2)) = (&self.matrix2d, &kernel.matrix2d){
            let m2d_out = m2d1.try_conv2(m2d2, kernel_shape, _strides, auto_pad, _pads, group, dilations)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let (Some(sub1), Some(sub2)) = (&self.sub_matrices, &kernel.sub_matrices){
            match self.get_dims().len() {
                4 => {
                    if self.dims[0] != 1 {
                        return Err(NotImplementedError)
                    }
                    let mut sub_out = Vec::new();
                    if sub1.len() == 1 && sub2.len() == 1{
                        sub_out.push(sub1[0].try_conv2(sub2[0].clone(), kernel_shape, _strides, auto_pad, _pads, group, dilations)?)
                    } else {
                        for sub2_slice in sub2.chunks(100){
                            let mut handles = Vec::new();
                            for s2 in sub2_slice.iter(){
                                let s1_new = sub1[0].clone();
                                let s2_new = s2.clone();
                                let kernel_shape_new = kernel_shape.clone();
                                let _strides_new = _strides.clone();
                                let auto_pads_new = auto_pad.clone();
                                let _pads_new = _pads.clone();
                                let group_new = group.clone();
                                let dilations_new = dilations.clone();
                                handles.push(thread::spawn(move || s1_new.try_conv2(s2_new, kernel_shape_new, _strides_new, auto_pads_new, _pads_new, group_new, dilations_new)));
                            }
                            for h in handles{
                                match h.join(){
                                    Ok(res) => sub_out.push(res?),
                                    Err(_) => return Err(ThreadingError)
                                }
                            };
                        };
                    }

                    // let sub_out = sub2.iter()
                    //     .map(|s2| sub1[0].try_conv(s2.clone(), kernel_shape.clone(), Arc::new(Some(_strides.clone())), auto_pad.clone(), Arc::new(Some(_pads.clone())), group.clone(), dilations.clone()))
                    //     .collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
                    let mut dims_out = sub_out[0].get_dims();
                    dims_out.insert(0, sub_out.len());
                    let mut matrix_out = Arc::new(Matrix::new_with_sub_matrices(dims_out.to_owned(), sub_out));
                    matrix_out = matrix_out.try_reshape(&vec![dims_out[1], dims_out[0], dims_out[2], dims_out[3]], None)?;
                    Ok(matrix_out)
                }
                3 => {
                    let mut sub_out = Vec::new();
                    if sub1.len() == 1 && sub2.len() == 1{
                        sub_out.push(sub1[0].try_conv2(sub2[0].clone(), kernel_shape, _strides, auto_pad, _pads, group, dilations)?)
                    } else {
                        let mut handles = Vec::new();
                        for (s1, s2) in sub1.iter().zip(sub2.iter()){
                            let s1_new = s1.clone();
                            let s2_new = s2.clone();
                            let kernel_shape_new = kernel_shape.clone();
                            let _strides_new = _strides.clone();
                            let auto_pads_new = auto_pad.clone();
                            let _pads_new = _pads.clone();
                            let group_new = group.clone();
                            let dilations_new = dilations.clone();
                            handles.push(thread::spawn(move || s1_new.try_conv2(s2_new, kernel_shape_new, _strides_new, auto_pads_new, _pads_new, group_new, dilations_new)));
                        }
                        for h in handles{
                            match h.join(){
                                Ok(res) => sub_out.push(res?),
                                Err(_) => return Err(ThreadingError)
                            }
                        };
                    }

                    // let mut sub_out = sub1.iter().zip(sub2.iter())
                    //     .map(|(s1, s2)| s1.try_conv(s2.clone(), kernel_shape.clone(), _strides.clone(), auto_pad.clone(), _pads.clone(), group.clone(), dilations.clone()))
                    //     .collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
                    if sub_out.len() > 1{
                        let mut start = sub_out[0].to_owned();
                        for  m in sub_out[1..].iter(){
                            start = start.try_add(m.clone())?;
                        }
                        sub_out = vec![start];
                    }
                    let mut dims_out = sub_out[0].get_dims();
                    dims_out.insert(0, sub_out.len());
                    Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
                }
                _ => {
                    Err(MatrixCompositionError)
                }
            }
            // let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_conv(s2, kernel_shape, Some(&_strides), auto_pad, Some(&_pads), None, None)).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            // let mut dims_out = sub_out[0].get_dims();
            // dims_out.insert(0, sub_out.len());
            // Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
        }
        else {
            Err(MatrixCompositionError)
        }

    }

    fn try_conv3(&self, kernel: Arc<Matrix<T>>, bias: Arc<Matrix<T>>, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        if bias.dims[0] != 1 || bias.dims[1] != kernel.dims[0]{
            return Err(MismatchSizeError);
        }
        let partial = self.try_conv2(kernel, kernel_shape, strides, auto_pad, pads, group, dilations)?;
        let data_out = bias.try_reshape(&vec![bias.dims[0], bias.dims[1], 1, 1], None)?;
        partial.try_add(data_out)
    }

    fn try_concat(&self, other: Arc<Matrix<T>>, axis: Arc<i64>) -> Result<Self::Output, MatrixOperationError> {
        let m1 = self.clone();
        let m2 = other;

        if ! m1.dims.iter().zip(m2.dims.iter()).enumerate().all(|(i, (d1, d2))| (i as i64) != *axis || d1 == d2){
            return Err(MismatchSizeError);
        }

        if let (Some(m2d1), Some(m2d2)) = (&m1.matrix2d, &m2.matrix2d){
            let m2d_out = m2d1.try_concat(m2d2, axis)?;
            Ok(Arc::new(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out)))
        }
        else if let (Some(sub1), Some(sub2)) = (&m1.sub_matrices, &m2.sub_matrices){
            let mut sub_out = Vec::new();
            if *axis == 0{
                sub1.iter().for_each(|s| sub_out.push(s.clone()));
                sub2.iter().for_each(|s| sub_out.push(s.clone()));
            } else {
                if sub1.len() == 1 && sub2.len() == 1{
                    sub_out.push(sub1[0].try_concat(sub2[0].clone(), Arc::new(*axis - 1))?)
                } else {
                    let mut handles = Vec::new();
                    for (s1, s2) in sub1.iter().zip(sub2.iter()){
                        let s1_new = s1.clone();
                        let s2_new = s2.clone();
                        let axis_new = Arc::new(*axis - 1);
                        handles.push(thread::spawn(move || s1_new.try_concat(s2_new, axis_new)));
                    }
                    for h in handles{
                        match h.join(){
                            Ok(res) => sub_out.push(res?),
                            Err(_) => return Err(ThreadingError)
                        }
                    };
                }
            }

            // let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_mat_mul(s2)).collect::<Result<Vec<Arc<Matrix<T>>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Arc::new(Matrix::new_with_sub_matrices(dims_out, sub_out)))
        }
        else {
            Err(MatrixCompositionError)
        }
    }
}

impl TryOperation2FloatOnly<Arc<Matrix<f32>>> for Arc<Matrix<f32>> {
    type Output = Arc<Matrix<f32>>;

    fn try_dropout(&self, ratio: Arc<Matrix<f32>>, seed: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError> {
        Ok(self.clone())
    }
}
#[derive(Debug, Clone)]
pub enum MatrixType{
    IntMatrix(Arc<Matrix<i64>>),
    FloatMatrix(Arc<Matrix<f32>>)
}

impl MatrixType{
    pub fn new(dims: Vec<usize>, ints: Option<Vec<i64>>, floats: Option<Vec<f32>>) -> Self{
        if ints.is_some(){
            IntMatrix(Arc::new(Matrix::new(dims, ints)))
        } else if floats.is_some(){
            FloatMatrix(Arc::new(Matrix::new(dims, floats)))
        } else {
            MatrixType::default()
        }
    }

    pub fn get_dims(&self) -> Vec<usize> {
        match &self {
            IntMatrix(matrix) => matrix.get_dims(),
            FloatMatrix(matrix) => matrix.get_dims()
        }
    }
}

impl Default for MatrixType {
    fn default() -> Self {
        MatrixType::IntMatrix(Arc::new(Matrix::default()))
    }
}

impl TryFrom<&TensorProto> for MatrixType {
    type Error = MatrixOperationError;

    fn try_from(tensor_proto: &TensorProto) -> Result<Self, Self::Error> {
        match tensor_proto.data_type.enum_value() {
            Ok(data_type) => {
                match data_type{
                    DataType::INT64 => Ok(
                        IntMatrix(Arc::new(Matrix::new(
                            tensor_proto.dims.iter().map(|d| *d as usize).collect(),
                            Some(tensor_proto.int64_data.to_owned())
                        )))
                    ),
                    DataType::FLOAT => Ok(
                        FloatMatrix(Arc::new(Matrix::new(
                            if tensor_proto.dims.len() == 0{
                                vec![1]
                            } else {
                                tensor_proto.dims.iter().map(|d| *d as usize).collect()
                            },
                            Some(
                                if tensor_proto.float_data.len() > 0{
                                    tensor_proto.float_data.to_owned()
                                } else {
                                    tensor_proto.raw_data.chunks_exact(4)
                                        .map(TryInto::try_into).map(Result::unwrap).map(f32::from_le_bytes)
                                        .collect::<Vec<f32>>()
                                }
                            )
                        )))
                    ),
                    _ => Err(NotImplementedError)
                }
            },
            Err(_) => Err(MismatchTypeError)
        }
    }
}

impl TryFrom<&ValueInfoProto> for MatrixType {
    type Error = MatrixOperationError;

    fn try_from(value_proto: &ValueInfoProto) -> Result<Self, Self::Error> {
        let Value::TensorType(tensor) = value_proto.type_.value.as_ref().unwrap();
        let dims = tensor.shape.dim.iter().filter_map(|d| {
            match &d.value {
                Some(value) => {
                    match value {
                        DimValue(v) => Some(Ok(*v as usize)),
                        DimParam(_) => Some(Err(NotImplementedError))
                    }
                },
                None => None
            }
        }).collect::<Result<Vec<usize>, MatrixOperationError>>()?;
        match tensor.elem_type.enum_value() {
            Ok(data_type) => {
                match data_type{
                    DataType::INT64 => Ok(
                        IntMatrix(Arc::new(Matrix::new(
                            dims,
                            None
                        )))
                    ),
                    DataType::FLOAT => Ok(
                        FloatMatrix(Arc::new(Matrix::new(
                            dims,
                            None
                        )))
                    ),
                    _ => Err(NotImplementedError)
                }
            },
            Err(_) => Err(MismatchTypeError)
        }
    }
}

impl TryOperation1 for MatrixType{
    type Output = Self;

    fn try_relu(&self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_relu()?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_relu()?))
        }
    }

    fn try_reshape(&self, dim: &Vec<usize>, allow_zero: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_reshape(dim, None)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_reshape(dim, None)?))
        }
    }

    fn try_broadcast(&self, dim: Arc<Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_broadcast(dim)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_broadcast(dim)?))
        }
    }

    fn try_max_pool(&self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, ceil_mode: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>, storage_order: Arc<Option<usize>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_max_pool(kernel_shape, strides, auto_pad, pads, ceil_mode, dilations, storage_order)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_max_pool(kernel_shape, strides, auto_pad, pads, ceil_mode, dilations, storage_order)?))
        }
    }

    fn try_global_max_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_global_max_pool()?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_global_max_pool()?))
        }
    }
}

impl TryOperation1FloatOnly for MatrixType {
    type Output = Self;

    fn try_softmax(&self, axis: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Err(MismatchTypeError),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_softmax(axis)?))
        }
    }

    fn try_global_average_pool(&self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Err(MismatchTypeError),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_global_average_pool()?))
        }
    }
}

impl TryOperation2<&Self> for MatrixType {
    type Output = Self;

    fn try_add(&self, other: &Self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_add(other_matrix.clone())?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_add(other_matrix.clone())?))
                }
            }
        }
    }

    fn try_mat_mul(&self, other: &Self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_mat_mul(other_matrix.clone())?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_mat_mul(other_matrix.clone())?))
                }
            }
        }
    }

    fn try_conv2(&self, kernel: &Self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match kernel {
                    IntMatrix(kernel_matrix) => Ok(IntMatrix(matrix.try_conv2(kernel_matrix.clone(), kernel_shape, strides, auto_pad, pads, group, dilations)?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match kernel {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(kernel_matrix) => Ok(FloatMatrix(matrix.try_conv2(kernel_matrix.clone(), kernel_shape, strides, auto_pad, pads, group, dilations)?))
                }
            }
        }
    }

    fn try_conv3(&self, kernel: &Self, bias: &Self, kernel_shape: Arc<Vec<usize>>, strides: Arc<Option<Vec<usize>>>, auto_pad: Arc<Option<String>>, pads: Arc<Option<Vec<usize>>>, group: Arc<Option<usize>>, dilations: Arc<Option<Vec<usize>>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                if let IntMatrix(kernel_matrix) = kernel{
                    if let IntMatrix(bias_matrix) = bias{
                        return Ok(IntMatrix(matrix.try_conv3(kernel_matrix.clone(), bias_matrix.clone(), kernel_shape, strides, auto_pad, pads, group, dilations)?));
                    }
                }
                Err(MismatchTypeError)
            },
            FloatMatrix(matrix) => {
                if let FloatMatrix(kernel_matrix) = kernel{
                    if let FloatMatrix(bias_matrix) = bias{
                        return Ok(FloatMatrix(matrix.try_conv3(kernel_matrix.clone(), bias_matrix.clone(), kernel_shape, strides, auto_pad, pads, group, dilations)?));
                    }
                }
                Err(MismatchTypeError)
            }
        }
    }

    fn try_concat(&self, other: &Self, axis: Arc<i64>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_concat(other_matrix.clone(), axis)?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_concat(other_matrix.clone(), axis)?))
                }
            }
        }
    }
}

impl TryOperation2FloatOnly<&Self> for MatrixType {
    type Output = Self;

    fn try_dropout(&self, ratio: &Self, seed: Arc<Option<i64>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                Err(MatrixOperationError::MismatchTypeError)
            },
            FloatMatrix(matrix) => {
                match ratio {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_dropout(other_matrix.clone(), seed)?))
                }
            }
        }
    }
}
impl TryOperation1Attributes for MatrixType{
    fn try_relu_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError> {
        self.try_relu()
    }

    fn try_max_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<Self, MatrixOperationError> {
        let mut kernel_shape: Vec<usize> = Vec::new();
        let mut strides: Option<Vec<usize>> = None;
        let mut auto_pad: Option<String> = None;
        let mut pads: Option<Vec<usize>> = None;
        let mut ceil_mode: Option<usize> = None;
        let mut dilations: Option<Vec<usize>> = None;
        let mut storage_order: Option<usize> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "kernel_shape" => kernel_shape = a.ints.iter().map(|i| *i as usize).collect(),
                "strides" => strides = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "auto_pad" => auto_pad = Some(String::from_utf8(a.s.to_owned()).unwrap()),
                "pads" => pads = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "ceil_mode" => ceil_mode = Some(a.i as usize),
                "dilations" => dilations = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "storage_order" => storage_order = Some(a.i as usize),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        if kernel_shape.len() == 0{
            Err(MissingFieldError)
        } else {
            self.try_max_pool(Arc::new(kernel_shape), Arc::new(strides), Arc::new(auto_pad), Arc::new(pads), Arc::new(ceil_mode), Arc::new(dilations), Arc::new(storage_order))
        }
    }

    fn try_global_max_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError> {
        self.try_global_max_pool()
    }

    fn try_global_average_pool_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError> {
        self.try_global_average_pool()
    }

    fn try_softmax_attributes(&self, attributes: &Vec<AttributeProto>) -> Result<MatrixType, MatrixOperationError> {
        let mut axis: Option<i64> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "axis" => axis = Some(a.i ),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        self.try_softmax(Arc::new(axis))
    }
}

impl TryOperation2Attributes for MatrixType {
    type Output = Self;

    fn try_add_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        self.try_add(other)
    }

    fn try_mat_mul_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        self.try_mat_mul(other)
    }

    fn try_conv2_attributes(&self, kernel: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        let mut kernel_shape: Vec<usize> = Vec::new();
        let mut strides: Option<Vec<usize>> = None;
        let mut auto_pad: Option<String> = None;
        let mut pads: Option<Vec<usize>> = None;
        let mut group: Option<usize> = None;
        let mut dilations: Option<Vec<usize>> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "kernel_shape" => kernel_shape = a.ints.iter().map(|i| *i as usize).collect(),
                "strides" => strides = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "auto_pad" => auto_pad = Some(String::from_utf8(a.s.to_owned()).unwrap()),
                "pads" => pads = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "group" => group = Some(a.i as usize),
                "dilations" => dilations = Some(a.ints.iter().map(|i| *i as usize).collect()),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        if kernel_shape.len() == 0{
            Err(MissingFieldError)
        } else {
            self.try_conv2(kernel, Arc::new(kernel_shape), Arc::new(strides), Arc::new(auto_pad), Arc::new(pads), Arc::new(group), Arc::new(dilations))
        }
    }

    fn try_conv3_attributes(&self, kernel: &Self, bias: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        let mut kernel_shape: Vec<usize> = Vec::new();
        let mut strides: Option<Vec<usize>> = None;
        let mut auto_pad: Option<String> = None;
        let mut pads: Option<Vec<usize>> = None;
        let mut group: Option<usize> = None;
        let mut dilations: Option<Vec<usize>> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "kernel_shape" => kernel_shape = a.ints.iter().map(|i| *i as usize).collect(),
                "strides" => strides = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "auto_pad" => auto_pad = Some(String::from_utf8(a.s.to_owned()).unwrap()),
                "pads" => pads = Some(a.ints.iter().map(|i| *i as usize).collect()),
                "group" => group = Some(a.i as usize),
                "dilations" => dilations = Some(a.ints.iter().map(|i| *i as usize).collect()),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        if kernel_shape.len() == 0{
            Err(MissingFieldError)
        } else {
            self.try_conv3(kernel, bias,Arc::new(kernel_shape), Arc::new(strides), Arc::new(auto_pad), Arc::new(pads), Arc::new(group), Arc::new(dilations))
        }
    }

    fn try_reshape_attributes(&self, dim: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        let mut allowzero: Option<usize> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "allowzero" => allowzero = Some(a.i as usize),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        match dim {
            IntMatrix(d) => {
                if d.dims.len() != 2 || d.dims[0] != 1{
                    return Err(MismatchSizeError)
                }
                let new_dim = d.get_data_or_error()?.iter().map(|dd| *dd as usize).collect();
                self.try_reshape(&new_dim, Option::from(&allowzero))
            },
            FloatMatrix(_) => Err(MismatchTypeError)
        }
    }

    fn try_dropout_attributes(&self, ratio: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        let mut seed: Option<i64> = None;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "seed" => seed = Some(a.i),
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        self.try_dropout(ratio, Arc::new(seed))
    }

    fn try_concat_attributes(&self, other: &Self, attributes: &Vec<AttributeProto>) -> Result<Self::Output, MatrixOperationError> {
        let mut axis: i64 = 0;
        let mut have_invalid_argument = false;
        attributes.iter().for_each(|a| {
            match a.name.as_str() {
                "axis" => axis = a.i,
                _ => have_invalid_argument = true
            }
        });
        if have_invalid_argument {
            return Err(InvalidArgumentError);
        }
        self.try_concat(other, Arc::new(axis))
    }
}