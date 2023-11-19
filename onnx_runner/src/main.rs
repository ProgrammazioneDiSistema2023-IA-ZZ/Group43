use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Add;
use std::string::String;
use onnx_runner::onnx::matrix::{Matrix, MatrixType, TryOperation1, TryOperation2};
use onnx_runner::parser::parser::Parser;
use onnx_runner::onnx::onnx_graph::OnnxGraph;

fn main() {
    println!("Welcome to the onnx runner!");
    // test_parser();
    //test_onnx();
    // test_tmp();
    // test_matrix();
    //verify_op();
    test_broadcast();
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
    let matrix_f = MatrixType::from(onnx_model.graph.initializer[2].to_owned());
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
    let m2 = m2.try_reshape(vec![2,100]);
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

fn test_broadcast(){
    fn add<T: Debug + Add<Output=T> + Copy>(m1: &Matrix<T>, m2: &Matrix<T>){
        println!("M1 => {:?}", m1);
        println!("M2 => {:?}", m2);
        let res = m1.try_add(&m2).unwrap();
        println!("RES => {:?}", res);
    }
    let m1 = Matrix::new(vec![2,3], Some(vec![1,2,3,4,5,6]));
    let m2 = Matrix::new(vec![2,3], Some(vec![2,2,2,2,2,2]));
    add(&m1, &m2);
}