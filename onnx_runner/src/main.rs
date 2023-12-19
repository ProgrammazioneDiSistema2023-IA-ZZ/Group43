use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::string::String;
use image::DynamicImage::ImageLuma8;
use onnx_runner::onnx::matrix::{Matrix, MatrixOperationError, MatrixType, Numeric, TryOperation1, TryOperation2};
use onnx_runner::parser::parser::Parser;
use onnx_runner::onnx::onnx_graph::OnnxGraph;
use image::io::Reader as ImageReader;
use onnx_runner::onnx::matrix::MatrixOperationError::MismatchTypeError;

fn main() {
    println!("Welcome to the onnx runner!");
    // test_parser();
    //test_onnx();
    // test_tmp();
    // test_matrix();
    //verify_op();
    // test_broadcast();
    // test_img();
    test_inference().unwrap();
}

fn test_parser() {
    println!("Test parser");
    let onnx_file = "bvlcalexnet-12-qdq";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file).expect("Error in onnx file parsing");
    Parser::store_to_json_file(onnx_file, & onnx_model).expect("Error in storing json");
    let _onnx_model_from_json = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
}

fn test_onnx() {
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::from(onnx_model);
    println!("{}", onnx);

    println!("{:?}", onnx.init_nodes[3].borrow().data);
}

fn test_tmp(){
    let onnx_file = "googlenet-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let result = onnx_model.graph.node.iter().all(|n| n.output.len() == 1);
    if result{
        println!("PASSED!");
    }else {
        println!("FAILED!");
    }
}

fn test_matrix() {
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    // let matrix_i = MatrixType::from(onnx_model.graph.initializer[5].to_owned());
    // println!("{:?}", matrix_i);
    // let matrix_f = MatrixType::from(onnx_model.graph.initializer[3].to_owned());
    // println!("{:?}", matrix_f);
    // let matrix_v = MatrixType::from(onnx_model.graph.input[0].to_owned());
    // println!("{:?}", matrix_v);
    let matrix_f = MatrixType::from(&onnx_model.graph.initializer[2]);
    println!("{:?}", matrix_f);
    // if let MatrixType::FloatMatrix(matrix) = matrix_f{
    //     let out = matrix.try_add(&matrix);
    //     println!("{:?}", out.unwrap());
    // }
    // if let MatrixType::FloatMatrix(matrix) = matrix_f{
    //     let out = matrix.try_relu();
    //     println!("{:?}", out.unwrap());
    // }
    let m = matrix_f.try_relu();
    println!("{:?}", m);
    let m2 = m.unwrap();
    let new_size = vec![2,100];
    let m2 = m2.try_reshape(&new_size, None);
    println!("{:?}", m2.unwrap());
}

// fn verify_op(){
//     let files_name = vec![/*"bvlcalexnet-12-qdq", "googlenet-12", "mnist-12", "mobilenetv2-12", "resnet18-v2-7",*/ "squeezenet1.0-12", "super-resolution-10"];
//     for name in files_name{
//         println!("{}", name);
//         let onnx_model = Parser::extract_from_json_file(name).expect("Error in json file parsing");
//         let onnx = OnnxGraph::from(onnx_model);
//         let mut op = HashSet::new();
//         for fun in onnx.fun_nodes{
//             op.insert(fun.borrow().get_operation_name().to_owned());
//         }
//         let mut op_out = op.iter().collect::<Vec<&String>>();
//         op_out.sort();
//         println!("{:?}", op_out);
//     }
// }

fn test_broadcast() {
    fn add<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
        println!("ADD");
        println!("M1 => {:?}", m1);
        println!("M2 => {:?}", m2);
        let res = m1.try_add(&m2).unwrap();
        println!("RES => {:?}", res);
    }
    let m1 = Matrix::new(vec![2, 1], Some(vec![1, 2]));
    let m2 = Matrix::new(vec![1, 3], Some(vec![2, 2, 2]));
    add(&m1, &m2);
    // let dim = vec![2,2,3];
    // println!("{:?}", m1);
    // let res = m1.try_broadcast(&dim).unwrap();
    // println!("{:?}", res);
    fn mul<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
        println!("MATMUL");
        println!("M1 => {:?}", m1);
        println!("M2 => {:?}", m2);
        let res = m1.try_mat_mul(&m2).unwrap();
        println!("RES => {:?}", res);
    }
    let m3 = Matrix::new(vec![2, 3], Some(vec![1, 0, 2, 0, 3, -1]));
    let m4 = Matrix::new(vec![3, 2], Some(vec![4, 1, -2, 2, 0, 3]));
    mul(&m3, &m4);
    let m5 = Matrix::new(vec![3, 3], Some(vec![1, 0, 1,1,5,-1,3,2,0]));
    let m6 = Matrix::new(vec![3, 2], Some(vec![7,1,1,0,0,4]));
    mul(&m5, &m6);
    // mul(&m4, &m5);
    fn maxpool<T: Numeric>(m1: &Matrix<T>) {
        println!("MAXPOOL");
        println!("M1 => {:?}", m1);
        let res = m1.try_max_pool(&vec![1,2], None, None, None, None, None, None).unwrap();
        println!("RES => {:?}", res);
    }
    let m7 = Matrix::new(vec![2, 3], Some(vec![1, 0, 2, 0, 3, -1]));
    maxpool(&m7);
    fn conv<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
        println!("CONV");
        println!("M1 => {:?}", m1);
        println!("M2 => {:?}", m2);
        let res = m1.try_conv(&m2, &vec![2, 2], Some(&vec![2,2]), None, Some(&vec![1,1,1,1]), None, None).unwrap();
        println!("RES => {:?}", res);
    }
    let m8 = Matrix::new(vec![3, 3], Some(vec![1, 0, 1,1,5,-1,3,2,0]));
    let m9 = Matrix::new(vec![2, 2], Some(vec![1, 0, 0, 1]));
    conv(&m8, &m9);
}

fn test_img(){
    let img = ImageReader::open("two.png").unwrap().decode().unwrap();
    println!("{:?}", img);
    if let ImageLuma8(luma) = img{
        let data = luma.iter().map(|p| (*p as f32) / 255.0).collect::<Vec<f32>>();
        println!("{:?}", data);
    }
}

fn test_inference() -> Result<(), MatrixOperationError>{
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::from(onnx_model);

    let img = ImageReader::open("two.png").unwrap().decode().unwrap();
    let input_matrix;
    match img {
        ImageLuma8(ref luma) => {
            let input_data = luma.iter().map(|p| (*p as f32) / 255.0).collect::<Vec<f32>>();
            input_matrix = MatrixType::new(vec![img.height() as usize, img.width() as usize], None, Some(input_data));
        },
        _ => return Err(MismatchTypeError)
    }
    onnx.try_load_data(input_matrix)?;
    println!("{:?}", onnx.root_node.borrow().data);

    let tmp = MatrixType::new(vec![2, 4], Some(vec![2,3,1,1,1,2,4,2]), None);
    println!("EASY MAXPOOL: {:?}", tmp.try_max_pool(&vec![2,2], Some(&vec![2,2]), None, None, None, None, None));
    println!("{:?}", onnx.fun_nodes[3].borrow().op1.is_some());
    let res = onnx.fun_nodes[3].borrow().calculate(tmp)?;
    println!("{:?}", res);
    Ok(())
}