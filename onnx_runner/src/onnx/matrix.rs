use std::error::Error;
use std::fmt::{Debug, Display, Formatter, write};
use std::ops::{Add, Mul};
use crate::onnx::matrix::MatrixType::{FloatMatrix, IntMatrix};
use crate::onnx::matrix::MatrixOperationError::{DontLastMatrixError, MismatchSizeError, VoidMatrixError, MatrixCompositionError, MismatchTypeError, NotImplementedError, MissingFieldError};
use crate::parser::onnx_model::onnx_proto3::{TensorProto, ValueInfoProto};
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
        }
    }
}

pub trait Numeric: Copy + Default + Debug + Add<Output = Self> + Mul<Output = Self> + PartialOrd{}

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

    fn try_broadcast(&self, dim: &Vec<usize>) -> Result<Self::Output, MatrixOperationError>;

    fn try_max_pool(&self, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, ceil_mode: Option<&usize>, dilations: Option<&Vec<usize>>, storage_order: Option<&usize>) -> Result<Self::Output, MatrixOperationError>;
}

pub trait TryOperation2<T>{
    type Output;

    fn try_add(&self, other: T) -> Result<Self::Output, MatrixOperationError>;

    fn try_mat_mul(&self, other: T) -> Result<Self::Output, MatrixOperationError>;

    fn try_conv(&self, kernel: T, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, group: Option<&usize>, dilations: Option<&Vec<usize>>) -> Result<Self::Output, MatrixOperationError>;
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

    fn try_broadcast(&self, dim: &Vec<usize>) -> Result<Self::Output, MatrixOperationError> {
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

    fn try_max_pool(&self, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, ceil_mode: Option<&usize>, dilations: Option<&Vec<usize>>, storage_order: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        let _strides = match strides {
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

    fn try_conv(&self, kernel: &Matrix2D<T>, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, group: Option<&usize>, dilations: Option<&Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        let _strides = match strides {
            Some(s) => s,
            None => return Err(MissingFieldError)
        };
        let _pads = match pads {
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
}

#[derive(Debug, Clone, Default)]
pub struct Matrix<T>{
    dims: Vec<usize>,
    matrix2d: Option<Matrix2D<T>>,
    sub_matrices: Option<Vec<Matrix<T>>>
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

    fn get_dim_for_broadcast(dim1: &Vec<usize>, dim2: &Vec<usize>) -> Result<Vec<usize>, MatrixOperationError>{
        let mut dim_same;
        let mut dim_enlarged;
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
    type Output = Matrix<T>;

    fn try_relu(&self) -> Result<Self::Output, MatrixOperationError> {
        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_relu()?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let Some(sub) = &self.sub_matrices{
            let sub_out = sub.iter().map(|s| s.try_relu()).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
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
        Ok(Matrix::new(dim.to_owned(), Some(data)))
    }

    fn try_broadcast(&self, dim: &Vec<usize>) -> Result<Self::Output, MatrixOperationError> {
        let this_dim = Self::enlarge_dim(&self.dims, dim.len());
        let out = self.try_reshape(&this_dim, None)?;
        if let Some(m2d) = out.matrix2d{
            let m2d_out = m2d.try_broadcast(&dim)?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let Some(subs) = out.sub_matrices {
            let mut sub_out;
            if out.dims[0] == 1{
                if out.dims[0] != dim[0]{
                    sub_out = (0..dim[0]).map(|_| subs[0].clone()).collect();
                }else {
                    sub_out = subs;
                }
            }else {
                if out.dims[0] != dim[0]{
                    return Err(MismatchSizeError);
                }else {
                    sub_out = subs;
                }
            }
            sub_out = sub_out.iter().map(|s| s.try_broadcast(&dim[1..].to_vec())).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_max_pool(&self, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, ceil_mode: Option<&usize>, dilations: Option<&Vec<usize>>, storage_order: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        if kernel_shape.len() != 2{
            return Err(NotImplementedError);
        }
        if let Some(a) = auto_pad {
            if a != "NOTSET"{
                return Err(NotImplementedError);
            }
        }
        if let Some(p) = pads {
            if p.iter().any(|pad| *pad != 0){
                return Err(NotImplementedError);
            }
        }
        if let Some(c) = ceil_mode{
            if *c != 0 {
                return Err(NotImplementedError);
            }
        }
        if let Some(d) = dilations{
            if d.iter().any(|dil| *dil != 1){
                return Err(NotImplementedError);
            }
        }
        if let Some(s) = storage_order{
            if *s != 0 {
                return Err(NotImplementedError);
            }
        }

        if let Some(m2d) = &self.matrix2d{
            let m2d_out = m2d.try_max_pool(kernel_shape, strides, auto_pad, pads, None, None, None)?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let Some(sub) = &self.sub_matrices{
            let sub_out = sub.iter().map(|s| s.try_max_pool(kernel_shape, strides, auto_pad, pads, None, None, None)).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
        }
        else {
            Err(VoidMatrixError)
        }
    }
}

impl<T: Numeric> TryOperation2<&Matrix<T>> for Matrix<T>{
    type Output = Matrix<T>;

    fn try_add(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixOperationError> {
        //if same len
        // => if same value
        //  => continue
        // => try broadcast either same dim
        //else
        // => create dim(reshape)
        // => try broadcast either to dim
        fn try_broadcast_either<T: Numeric>(this: &Matrix<T> , other: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>), MatrixOperationError>{
            let broadcast_dim = Matrix::<T>::get_dim_for_broadcast(&this.dims, &other.dims)?;
            let m1 = this.try_broadcast(&broadcast_dim)?;
            let m2 = other.try_broadcast(&broadcast_dim)?;
            Ok((m1, m2))
        }
        let m1;
        let m2;
        let mut m1_ref = self;
        let mut m2_ref = other;
        if self.dims.len() == other.dims.len(){
            if self.dims.iter().zip(other.dims.iter()).any(|(d1, d2)| d1 != d2){
                (m1, m2) = try_broadcast_either(self, other)?;
                m1_ref = &m1;
                m2_ref = &m2;
            }
        }else{
            (m1, m2) = try_broadcast_either(self, other)?;
            m1_ref = &m1;
            m2_ref = &m2;
        }

        if let (Some(m2d1), Some(m2d2)) = (&m1_ref.matrix2d, &m2_ref.matrix2d){
            let m2d_out = m2d1.try_add(m2d2)?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let (Some(sub1), Some(sub2)) = (&m1_ref.sub_matrices, &m2_ref.sub_matrices){
            let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_add(s2)).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_mat_mul(&self, other: &Matrix<T>) -> Result<Self::Output, MatrixOperationError> {
        let mut m1_ref = self;
        let mut m2_ref = other;

        if let (Some(m2d1), Some(m2d2)) = (&m1_ref.matrix2d, &m2_ref.matrix2d){
            let m2d_out = m2d1.try_mat_mul(m2d2)?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let (Some(sub1), Some(sub2)) = (&m1_ref.sub_matrices, &m2_ref.sub_matrices){
            let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_mat_mul(s2)).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
        }
        else {
            Err(MatrixCompositionError)
        }
    }

    fn try_conv(&self, kernel: &Matrix<T>, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, group: Option<&usize>, dilations: Option<&Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        fn get_pad_couple(dim: usize, kernel_dim: usize, stride_dim: usize, is_upper: bool) -> Vec<usize>{
            let out_dim = (dim + stride_dim - 1) / stride_dim;
            let total_pads = kernel_dim - dim + stride_dim * (out_dim - 1);
            let small_pad = total_pads / 2;
            let big_pad = if total_pads%2 == 0 { small_pad } else { small_pad+1 };
            if is_upper{
                vec![small_pad, big_pad]
            } else {
                vec![big_pad, small_pad]
            }
        }

        if kernel_shape.len() != 2{
            return Err(NotImplementedError);
        }
        if let Some(g) = group{
            if *g != 0 {
                return Err(NotImplementedError);
            }
        }
        if let Some(d) = dilations{
            if d.iter().any(|dil| *dil != 1){
                return Err(NotImplementedError);
            }
        }

        let _strides: Vec<usize> = match strides {
            Some(s) => s.to_owned(),
            None => vec![1; kernel_shape.len()]
        };
        let _pads: Vec<usize> = match auto_pad {
            Some(a_p) => {
                match a_p.as_str() {
                    "NOTSET" => {
                        if let Some(p) = pads{
                            p.into_iter().map(|val| *val).collect()
                        } else {
                            vec![0; 4]
                        }
                    }
                    "SAME_UPPER" => {
                        let mut row_pad = get_pad_couple(self.dims[self.dims.len()-2], kernel.dims[kernel.dims.len()-2], _strides[0], true);
                        let mut col_pad = get_pad_couple(self.dims[self.dims.len()-1], kernel.dims[kernel.dims.len()-1], _strides[1], true);
                        row_pad.append(&mut col_pad);
                        row_pad
                    }
                    "SAME_LOWER" => {
                        let mut row_pad = get_pad_couple(self.dims[self.dims.len()-2], kernel.dims[kernel.dims.len()-2], _strides[0], false);
                        let mut col_pad = get_pad_couple(self.dims[self.dims.len()-1], kernel.dims[kernel.dims.len()-1], _strides[1], false);
                        row_pad.append(&mut col_pad);
                        row_pad
                    }
                    _ => return Err(NotImplementedError)
                }
            }
            None => {
                if let Some(p) = pads{
                    p.into_iter().map(|val| *val).collect()
                } else {
                    vec![0; 4]
                }
            }
        };

        if let (Some(m2d1), Some(m2d2)) = (&self.matrix2d, &kernel.matrix2d){
            let m2d_out = m2d1.try_conv(m2d2, kernel_shape, Some(&_strides), auto_pad, Some(&_pads), None, None)?;
            Ok(Matrix::new_with_matrix2d(m2d_out.get_dims(), m2d_out))
        }
        else if let (Some(sub1), Some(sub2)) = (&self.sub_matrices, &kernel.sub_matrices){
            let sub_out = sub1.iter().zip(sub2.iter()).map(|(s1, s2)| s1.try_conv(s2, kernel_shape, Some(&_strides), auto_pad, Some(&_pads), None, None)).collect::<Result<Vec<Matrix<T>>, MatrixOperationError>>()?;
            let mut dims_out = sub_out[0].get_dims();
            dims_out.insert(0, sub_out.len());
            Ok(Matrix::new_with_sub_matrices(dims_out, sub_out))
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

impl Default for MatrixType {
    fn default() -> Self {
        MatrixType::IntMatrix(Matrix::default())
    }
}

impl From<&TensorProto> for MatrixType {
    fn from(tensor_proto: &TensorProto) -> Self {
        match tensor_proto.data_type.enum_value() {
            Ok(data_type) => {
                match data_type{
                    DataType::INT64 => IntMatrix(Matrix::new(
                        tensor_proto.dims.iter().map(|d| *d as usize).collect(),
                        Some(tensor_proto.int64_data.to_owned())
                    )),
                    DataType::FLOAT => FloatMatrix(Matrix::new(
                        tensor_proto.dims.iter().map(|d| *d as usize).collect(),
                        Some(tensor_proto.float_data.to_owned())
                    )),
                    _ => panic!("Not supported data type")
                }
            },
            Err(_) => panic!("No valid data type")
        }
    }
}

impl From<&ValueInfoProto> for MatrixType {
    fn from(value_proto: &ValueInfoProto) -> Self {
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

impl<'a, T: Numeric> Data<'a, T> for MatrixType {
    type Output = Vec<T>;

    fn get_data_or_error(&'a self) -> Result<Self::Output, MatrixOperationError> {
        Err(NotImplementedError)
    }

    fn get_dims(&self) -> Vec<usize> {
        match &self {
            IntMatrix(matrix) => matrix.get_dims(),
            FloatMatrix(matrix) => matrix.get_dims()
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

    fn try_broadcast(&self, dim: &Vec<usize>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_broadcast(dim)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_broadcast(dim)?))
        }
    }

    fn try_max_pool(&self, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, ceil_mode: Option<&usize>, dilations: Option<&Vec<usize>>, storage_order: Option<&usize>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => Ok(IntMatrix(matrix.try_max_pool(kernel_shape, strides, auto_pad, pads, None, None, None)?)),
            FloatMatrix(matrix) => Ok(FloatMatrix(matrix.try_max_pool(kernel_shape, strides, auto_pad, pads, None, None, None)?))
        }
    }
}

impl TryOperation2<&Self> for MatrixType {
    type Output = Self;

    fn try_add(&self, other: &Self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_add(other_matrix)?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_add(other_matrix)?))
                }
            }
        }
    }

    fn try_mat_mul(&self, other: &Self) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match other {
                    IntMatrix(other_matrix) => Ok(IntMatrix(matrix.try_mat_mul(other_matrix)?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match other {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(other_matrix) => Ok(FloatMatrix(matrix.try_mat_mul(other_matrix)?))
                }
            }
        }
    }

    fn try_conv(&self, kernel: &Self, kernel_shape: &Vec<usize>, strides: Option<&Vec<usize>>, auto_pad: Option<&String>, pads: Option<&Vec<usize>>, group: Option<&usize>, dilations: Option<&Vec<usize>>) -> Result<Self::Output, MatrixOperationError> {
        match &self {
            IntMatrix(matrix) => {
                match kernel {
                    IntMatrix(kernel_matrix) => Ok(IntMatrix(matrix.try_conv(kernel_matrix, kernel_shape, strides, auto_pad, pads, group, dilations)?)),
                    FloatMatrix(_) => Err(MatrixOperationError::MismatchTypeError)
                }
            },
            FloatMatrix(matrix) => {
                match kernel {
                    IntMatrix(_) => Err(MatrixOperationError::MismatchSizeError),
                    FloatMatrix(kernel_matrix) => Ok(FloatMatrix(matrix.try_conv(kernel_matrix, kernel_shape, strides, auto_pad, pads, group, dilations)?))
                }
            }
        }
    }
}