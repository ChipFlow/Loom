use gem::liberty_parser::TimingLibrary;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: liberty_bench <path>");

    println!("Reading file: {}", path);
    let start = Instant::now();
    let content = std::fs::read_to_string(&path).expect("Failed to read file");
    println!("Read {} bytes in {:?}", content.len(), start.elapsed());

    println!("Parsing liberty file...");
    let start = Instant::now();
    match TimingLibrary::parse(&content) {
        Ok(lib) => {
            let elapsed = start.elapsed();
            println!("Parsed in {:?}", elapsed);
            println!("Library: {}", lib.name);
            println!("Cells: {}", lib.cells.len());
        }
        Err(e) => {
            println!("Parse error after {:?}: {}", start.elapsed(), e);
        }
    }
}
