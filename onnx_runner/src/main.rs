use std::error::Error;
use image::DynamicImage::{ImageLuma8, ImageRgb8};
use image::io::Reader as ImageReader;
use onnx_runner::onnx::matrix::{Data, MatrixOperationError, MatrixType};
use onnx_runner::onnx::matrix::MatrixOperationError::MismatchTypeError;
use onnx_runner::onnx::matrix::MatrixType::FloatMatrix;
use onnx_runner::onnx::onnx_graph::OnnxGraph;
use onnx_runner::parser::parser::Parser;

fn main() {
    println!("Welcome to the onnx runner!");

    demo_parser().unwrap();
    demo_mnist().unwrap();
    demo_squeezenet().unwrap();
}

fn demo_parser() -> Result<(), Box<dyn Error>> {
    println!("Demo parser:");
    let onnx_file = "mnist-12";
    let onnx_model = Parser::extract_from_onnx_file(onnx_file)?;
    Parser::store_to_json_file(onnx_file, & onnx_model)?;
    println!("{} model json stored successfully!\n", onnx_file);
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
            println!("Expected value {} => Prediction {} (max: {})\n", v, max_id, max);
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