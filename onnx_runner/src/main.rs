use std::collections::HashSet;
use std::error::Error;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::string::String;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use image::DynamicImage::{ImageLuma8, ImageRgb8};
use onnx_runner::onnx::matrix::{Data, Matrix, MatrixOperationError, MatrixType, Numeric, TryOperation1, TryOperation1FloatOnly, TryOperation2, TryOperation2FloatOnly};
use onnx_runner::parser::parser::Parser;
use onnx_runner::onnx::onnx_graph::OnnxGraph;
use image::io::Reader as ImageReader;
use onnx_runner::onnx::matrix::MatrixOperationError::MismatchTypeError;
use onnx_runner::onnx::matrix::MatrixType::FloatMatrix;
use onnx_runner::onnx::onnx_node::HaveOut;

fn main() {
    println!("Welcome to the onnx runner!");
    // test_parser();
    //test_onnx();
    //test_tmp();
    // test_matrix();
    //verify_op();
    // test_broadcast();
    // test_img();
    // test_inference().unwrap();
    //test_new_op();
    // test_squeeze().unwrap();

    demo_parser().unwrap();
    // demo_mnist().unwrap();
    demo_squeezenet().unwrap();
}

fn demo_parser() -> Result<(), Box<dyn Error>> {
    println!("Demo parser:");
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file)?;
    Parser::store_to_json_file(onnx_file, & onnx_model)?;
    println!("{} model json stored successfully!", onnx_file);
    Ok(())
}

fn demo_mnist() -> Result<(), MatrixOperationError>{
    println!("Demo mnist:");
    let onnx_file = "mnist-12";
    let values = (0..10).map(|v| v).collect::<Vec<i32>>();

    for v in values{
        let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
        let onnx = OnnxGraph::try_from(onnx_model)?;
        let img = ImageReader::open(v.to_string() + ".png").unwrap().decode().unwrap();
        let input_matrix;
        match img {
            ImageLuma8(ref luma) => {
                let input_data = luma.iter().map(|p| (*p as f32) / 255.0).collect::<Vec<f32>>();
                input_matrix = MatrixType::new(vec![img.height() as usize, img.width() as usize], None, Some(input_data));
            },
            _ => return Err(MismatchTypeError)
        }
        onnx.try_load_data(input_matrix)?;
        let mut out_node = onnx.output_nodes[0].borrow_mut();
        let out_matrix = out_node.try_compute_all()?;
        if let FloatMatrix(out) = out_matrix {
            let out_data = out.get_data_or_error()?;
            println!("Result {:?}", out_data);
            let (max_id, max) = out_data.iter().enumerate().fold((0, out_data[0]), |(id_m_acc, m_acc), (id_m, m)| {
                if m_acc > *m {
                    (id_m_acc, m_acc)
                } else {
                    (id_m, *m)
                }
            });
            println!("Expected value {} => Prediction {}\n", v, max_id);
        }
    }
    Ok(())
}

fn demo_squeezenet() -> Result<(), MatrixOperationError>{
    let onnx_file = "squeezenet1.0-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::try_from(onnx_model)?;

    let img = ImageReader::open("zebra_white.png").unwrap().decode().unwrap();

    let input_matrix;
    match img {
        ImageRgb8(ref rgb) => {
            let mut input_data = Vec::new();
            rgb.pixels().map(|p| ((p.0[0] as f32) / 255.0 - 0.485) / 0.229).for_each(|d| input_data.push(d));
            rgb.pixels().map(|p| ((p.0[1] as f32) / 255.0 - 0.456) / 0.224).for_each(|d| input_data.push(d));
            rgb.pixels().map(|p| ((p.0[2] as f32) / 255.0 - 0.406) / 0.225).for_each(|d| input_data.push(d));

            input_matrix = MatrixType::new(vec![1, 3, img.height() as usize, img.width() as usize], None, Some(input_data));
        },
        _ => return Err(MismatchTypeError)
    }
    onnx.try_load_data(input_matrix)?;

    let mut out_node = onnx.output_nodes[0].borrow_mut();
    let out_matrix = out_node.try_compute_all()?;
    if let FloatMatrix(out) = out_matrix {
        let out_data = out.get_data_or_error()?;
        // println!("Result {:?}", out_data);
        let (max_id, max) = out_data.iter().enumerate().fold((0, out_data[0]), |(id_m_acc, m_acc), (id_m, m)| {
            if m_acc > *m {
                (id_m_acc, m_acc)
            } else {
                (id_m, *m)
            }
        });
        println!("Expected value 669 car => Prediction {} (max: {})\n", max_id, max);
    }

    Ok(())
}

fn test_parser() {
    println!("Test parser");
    let onnx_file = "bvlcalexnet-12-qdq";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file).expect("Error in onnx file parsing");
    Parser::store_to_json_file(onnx_file, & onnx_model).expect("Error in storing json");
    let _onnx_model_from_json = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
}

fn test_onnx() {
    let onnx_file = "squeezenet1.0-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::try_from(onnx_model);
    // println!("{:?}", onnx);
    if let Ok(o) = onnx{
        println!("{:?}", o.init_nodes[0].borrow().data)
    }
    // println!("{:?}", onnx.init_nodes[3].borrow().data);
}

fn test_tmp(){
    let onnx_file = "googlenet-12";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file).expect("Error in onnx file parsing");
    Parser::store_to_json_file(onnx_file, & onnx_model).expect("Error in storing json");
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

fn verify_op(){
    let files_name = vec!["candy-9"];
    for name in files_name{
        println!("{}", name);
        let onnx_model = Parser::extract_from_onnx_file(name).expect("Error in onnx file parsing");
        Parser::store_to_json_file(name, & onnx_model).expect("Error in storing json");
        let onnx_model = Parser::extract_from_json_file(name).expect("Error in json file parsing");
        let onnx = OnnxGraph::try_from(onnx_model).expect("damn");
        let mut op = HashSet::new();
        for fun in onnx.fun_nodes{
            op.insert(fun.borrow().get_operation_name().to_owned());
        }
        let mut op_out = op.iter().collect::<Vec<&String>>();
        op_out.sort();
        println!("{:?}", op_out);
    }
}

fn test_new_op(){
    fn globalmp<T: Numeric>(m1: &Matrix<T>) {
        println!("glodal max pool");
        println!("M1 => {:?}", m1);
        let res = m1.try_global_max_pool().unwrap();
        println!("RES => {:?}", res);
    }
    let m1 = Matrix::new(vec![1, 2, 2, 3], Some(vec![1, 0, 2, 0, 3, -1, 1, 0, 5, 0, 3, -8]));
    globalmp(&m1);

    fn softmax(m1: &Matrix<f32>) {
        let m1 = Arc::new(m1.to_owned());
        println!("softmax");
        println!("M1 => {:?}", m1);
        let res = m1.try_softmax(Arc::new(None)).unwrap();
        println!("RES => {:?}", res);
    }
    let m2 = Matrix::new(vec![2, 3], Some(vec![1.1, 0.1, 2.1, 0.1, 3.1, -1.1]));
    softmax(&m2);

    fn dropout(m1: &Matrix<f32>) {
        let m1 = Arc::new(m1.to_owned());
        println!("dropout");
        println!("M1 => {:?}", m1);
        let res = m1.try_dropout(Arc::new(Matrix::default()), Arc::new(None)).unwrap();
        println!("RES => {:?}", res);
    }
    let m2 = Matrix::new(vec![2, 3], Some(vec![1.1, 0.1, 2.1, 0.1, 3.1, -1.1]));
    dropout(&m2);
    fn cat<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
        let m1 = Arc::new(m1.to_owned());
        let m2 = Arc::new(m2.to_owned());

        println!("CONCAT");
        println!("M1 => {:?}", m1);
        println!("M2 => {:?}", m2);
        let res = m1.try_concat(m2, Arc::new(1)).unwrap();
        println!("RES => {:?}", res);
    }
    let m3 = Matrix::new(vec![1, 1, 2, 3], Some(vec![1, 0, 2, 0, 3, -1]));
    let m4 = Matrix::new(vec![1, 1, 2, 3], Some(vec![4, 1, -2, 2, 0, 3]));
    cat(&m3, &m4);

    let mm = Matrix::new(vec![1,4], Some(vec![1,2,3,4]));
    println!("{:?}", mm);
    let mmm = mm.try_reshape(&vec![1,4,1,1], None).unwrap();
    println!("{:?}", mmm);
    let mmmm = mmm.try_broadcast(Arc::new(vec![1,4,2,2]));
    println!("{:?}", mmmm);

}

// fn test_broadcast() {
//     fn add<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
//         println!("ADD");
//         println!("M1 => {:?}", m1);
//         println!("M2 => {:?}", m2);
//         let res = m1.try_add(&m2).unwrap();
//         println!("RES => {:?}", res);
//     }
//     let m1 = Matrix::new(vec![2, 1], Some(vec![1, 2]));
//     let m2 = Matrix::new(vec![1, 3], Some(vec![2, 2, 2]));
//     add(&m1, &m2);
//     // let dim = vec![2,2,3];
//     // println!("{:?}", m1);
//     // let res = m1.try_broadcast(&dim).unwrap();
//     // println!("{:?}", res);
//     fn mul<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
//         println!("MATMUL");
//         println!("M1 => {:?}", m1);
//         println!("M2 => {:?}", m2);
//         let res = m1.try_mat_mul(&m2).unwrap();
//         println!("RES => {:?}", res);
//     }
//     let m3 = Matrix::new(vec![2, 3], Some(vec![1, 0, 2, 0, 3, -1]));
//     let m4 = Matrix::new(vec![3, 2], Some(vec![4, 1, -2, 2, 0, 3]));
//     mul(&m3, &m4);
//     let m5 = Matrix::new(vec![3, 3], Some(vec![1, 0, 1,1,5,-1,3,2,0]));
//     let m6 = Matrix::new(vec![3, 2], Some(vec![7,1,1,0,0,4]));
//     mul(&m5, &m6);
//     // mul(&m4, &m5);
//     fn maxpool<T: Numeric>(m1: &Matrix<T>) {
//         println!("MAXPOOL");
//         println!("M1 => {:?}", m1);
//         let res = m1.try_max_pool(&vec![1,2], None, None, None, None, None, None).unwrap();
//         println!("RES => {:?}", res);
//     }
//     let m7 = Matrix::new(vec![2, 3], Some(vec![1, 0, 2, 0, 3, -1]));
//     maxpool(&m7);
//     fn conv<T: Numeric>(m1: &Matrix<T>, m2: &Matrix<T>) {
//         println!("CONV");
//         println!("M1 => {:?}", m1);
//         println!("M2 => {:?}", m2);
//         let res = m1.try_conv(&m2, &vec![2, 2], Some(&vec![2,2]), None, Some(&vec![1,1,1,1]), None, None).unwrap();
//         println!("RES => {:?}", res);
//     }
//     let m8 = Matrix::new(vec![3, 3], Some(vec![1, 0, 1,1,5,-1,3,2,0]));
//     let m9 = Matrix::new(vec![2, 2], Some(vec![1, 0, 0, 1]));
//     conv(&m8, &m9);
// }

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
    let onnx = OnnxGraph::try_from(onnx_model)?;

    let img = ImageReader::open("seven.png").unwrap().decode().unwrap();
    let input_matrix;
    match img {
        ImageLuma8(ref luma) => {
            let input_data = luma.iter().map(|p| (*p as f32) / 255.0).collect::<Vec<f32>>();
            input_matrix = MatrixType::new(vec![img.height() as usize, img.width() as usize], None, Some(input_data));
        },
        _ => return Err(MismatchTypeError)
    }
    onnx.try_load_data(input_matrix)?;
    // println!("{:?}", onnx.root_node.borrow().data);

    // let tmp = MatrixType::new(vec![2, 4], Some(vec![2,3,1,1,1,2,4,2]), None);
    // println!("EASY MAXPOOL: {:?}", tmp.try_max_pool(&vec![2,2], Some(&vec![2,2]), None, None, None, None, None));
    // println!("{:?}", onnx.fun_nodes[3].borrow().op1.is_some());
    // let res = onnx.fun_nodes[3].borrow().calculate(tmp)?;
    // println!("{:?}", res);

    println!("All fun node have op: {:?}", onnx.fun_nodes.iter().all(|n| n.borrow().op1.is_some() || n.borrow().op2.is_some()));

    println!("RootData {:?}", onnx.root_node.borrow_mut().try_calculate());
    println!("Conv1Data {:?}", onnx.fun_nodes[0].borrow_mut().try_calculate());
    println!("Add1Data {:?}", onnx.fun_nodes[1].borrow_mut().try_calculate());
    println!("Relu1Data {:?}", onnx.fun_nodes[2].borrow_mut().try_calculate());
    println!("MaxPool1Data {:?}", onnx.fun_nodes[3].borrow_mut().try_calculate());
    println!("Conv2Data {:?}", onnx.fun_nodes[4].borrow_mut().try_calculate());
    println!("Add2Data {:?}", onnx.fun_nodes[5].borrow_mut().try_calculate());
    println!("Relu2Data {:?}", onnx.fun_nodes[6].borrow_mut().try_calculate());
    println!("MaxPool2Data {:?}", onnx.fun_nodes[7].borrow_mut().try_calculate());
    println!("ReshapeData {:?}", onnx.fun_nodes[8].borrow_mut().try_calculate());
    println!("ReshapeAltData {:?}", onnx.fun_nodes[9].borrow_mut().try_calculate());
    println!("MatMul2Data {:?}", onnx.fun_nodes[10].borrow_mut().try_calculate());
    println!("AddData {:?}", onnx.fun_nodes[11].borrow_mut().try_calculate());
    // println!("Out {:?}", onnx.output_nodes[0].borrow_mut().try_compute_all());

    Ok(())
}

fn test_squeeze() -> Result<(), MatrixOperationError>{
    let onnx_file = "squeezenet1.0-12";
    let onnx_model = Parser::extract_from_json_file(onnx_file).expect("Error in json file parsing");
    let onnx = OnnxGraph::try_from(onnx_model)?;

    let img = ImageReader::open("tench_resized.png").unwrap().decode().unwrap();
    let input_matrix;
    match img {
        ImageRgb8(ref rgb) => {
            let mut input_data = Vec::new();
            rgb.pixels().map(|p| ((p.0[0] as f32) / 255.0)).for_each(|d| input_data.push(d));
            rgb.pixels().map(|p| ((p.0[1] as f32) / 255.0)).for_each(|d| input_data.push(d));
            rgb.pixels().map(|p| ((p.0[2] as f32) / 255.0)).for_each(|d| input_data.push(d));
            input_matrix = MatrixType::new(vec![1, 3, img.height() as usize, img.width() as usize], None, Some(input_data));
        },
        _ => return Err(MismatchTypeError)
    }
    println!("hi {:?}", input_matrix);

    // println!("In mat {:?}", input_matrix);
    onnx.try_load_data(input_matrix)?;
    // println!("-1 {:?}", onnx.root_node.borrow_mut().try_calculate().unwrap().get_dims());
    // println!("0 {:?}", onnx.fun_nodes[0].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("rel {:?}", onnx.fun_nodes[4].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("concat {:?}", onnx.fun_nodes[9].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("rel {:?}", onnx.fun_nodes[11].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("concat {:?}", onnx.fun_nodes[16].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("rel {:?}", onnx.fun_nodes[19].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("concat {:?}", onnx.fun_nodes[24].borrow_mut().try_calculate().unwrap().get_dims());


    for (i, n) in onnx.fun_nodes.iter().enumerate(){
        let mut nn = n.borrow_mut();
        if nn.get_operation_name() == "Relu"{
            println!("{} {:?}", i, nn.try_calculate().unwrap().get_dims());
        }
        if i == 59{
            break;
        }
    }
    // println!("conc {:?}", onnx.fun_nodes[60].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("drop {:?}", onnx.fun_nodes[61].borrow_mut().try_calculate().unwrap().get_dims());
    // println!("conv {:?}", onnx.fun_nodes[62].borrow_mut().try_calculate().unwrap().get_dims());

    let mut br = onnx.output_nodes[0].borrow_mut();
    let out = br.try_compute_all();
    // println!("Out {:?}", out);
    if let FloatMatrix(fl) = out.unwrap() {
        let data = fl.get_data_or_error()?;
        let mut max = data[0];
        let mut max_id = 0;
        data.iter().enumerate().for_each(|(i, d)| {
            if *d > max {
                max = *d;
                max_id = i;
            }
        });
        for i in 0..100{
            println!("{}", data[i]);
        }
        println!("{} {}", max_id, max);
    }
    Ok(())
}