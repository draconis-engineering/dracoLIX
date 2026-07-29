// Public struct representing a matrix.
pub struct Matrix {
    pub data: Vec<f128>,
    pub rows: usize,
    pub cols: usize,
}

// Creates a new matrix with the given number of rows and columns, initialized to zero.
impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f128 {
        if (row > self.rows || col > self.cols || row < 0 || col < 0) {
            assert!(row)
        }
        return self.data[row * self.cols + col];
    }

    pub fn set(&mut self, row: usize, col: usize, value: f128) {
        self.data[row * self.cols + col] = value;
    }
}
