// struct Tensor{
//     dims: Vec<u64>,
//     uint: Vec<u64>,
//     int: Vec<i64>,
//     float: Vec<f64>,
//     raw: Vec<u8>,
//     string: Vec<u8>
// }
struct Matrix2D{
    rows: u64,
    cols: u64,
    data: Vec<i64>
}

impl Matrix2D {
    fn new(rows: u64, cols: u64, data: Vec<i64>) -> Self{
        Matrix2D{
            rows,
            cols,
            data
        }
    }
}

pub struct Matrix{
    dims: Vec<u64>,
    data: Option<Matrix2D>,
    sub_matrices: Option<Vec<Matrix>>
}

impl Matrix {
    fn new(dims: Vec<u64>, data: Vec<i64>) -> Self{
        match dims.len() {
            1 => Matrix{
                    dims: vec![1, dims[0]],
                    data: Some(Matrix2D::new(1, dims[0], data)),
                    sub_matrices: None
                },
            2 => {
                let rows = dims[0];
                let cols = dims[1];
                Matrix {
                    dims: dims,
                    data: Some(Matrix2D::new(rows, cols, data)),
                    sub_matrices: None,
                }
            },
            _ => {
                let sub_dims = &dims[1..];
                let sub_data_count = sub_dims.iter().fold(1, |acc, val| acc * val);
                let mut sub_matrices = Vec::new();
                for i in 0..dims[0] {
                    sub_matrices.push(Matrix::new(Vec::from(sub_dims), data[(i*sub_data_count) as usize..((i+1)*sub_data_count) as usize].to_owned()));
                }
                Matrix{
                    dims: dims,
                    data: None,
                    sub_matrices: Some(sub_matrices)
                }
            }
        }
    }
}
