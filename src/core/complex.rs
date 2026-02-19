/// A complex number with 64-bit floating-point components.
///
/// Represents amplitudes in quantum state vectors.
/// All quantum probability amplitudes are complex numbers α + βi
/// where |α|² + |β|² contributes to total probability.
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// Imaginary unit i
    #[inline(always)]
    pub const fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    /// Squared magnitude: |z|² = re² + im²
    #[inline(always)]
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude: |z| = sqrt(re² + im²)
    #[inline(always)]
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Complex conjugate: z* = re - im·i
    #[inline(always)]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    #[inline(always)]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Scale by a real scalar
    #[inline(always)]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Check near-zero within epsilon
    #[inline(always)]
    pub fn is_zero(&self, epsilon: f64) -> bool {
        self.norm_sq() < epsilon * epsilon
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
impl Mul for Complex {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < 1e-10 && (self.im - other.im).abs() < 1e-10
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6} + {:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6} - {:.6}i", self.re, self.im.abs())
        }
    }
}

impl From<f64> for Complex {
    fn from(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        assert_eq!(a + b, Complex::new(4.0, 6.0));
    }

    #[test]
    fn test_sub() {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(2.0, 1.0);
        assert_eq!(a - b, Complex::new(3.0, 2.0));
    }

    #[test]
    fn test_mul() {
        // (1 + 2i)(3 + 4i) = (3 - 8) + (4 + 6)i = -5 + 10i
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        assert_eq!(a * b, Complex::new(-5.0, 10.0));
    }

    #[test]
    fn test_mul_i_squared() {
        // i * i = -1
        let i = Complex::i();
        assert_eq!(i * i, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn test_conj() {
        let z = Complex::new(3.0, -4.0);
        assert_eq!(z.conj(), Complex::new(3.0, 4.0));
    }

    #[test]
    fn test_norm() {
        // |3 + 4i| = 5
        let z = Complex::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_sq() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.norm_sq() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_polar() {
        // e^(iπ) = -1 + 0i  (Euler's identity)
        let z = Complex::from_polar(1.0, PI);
        assert!((z.re - (-1.0)).abs() < 1e-10);
        assert!(z.im.abs() < 1e-10);
    }

    #[test]
    fn test_div() {
        // (1 + 2i) / (1 + 2i) = 1
        let z = Complex::new(1.0, 2.0);
        assert_eq!(z / z, Complex::one());
    }

    #[test]
    fn test_neg() {
        let z = Complex::new(1.0, -2.0);
        assert_eq!(-z, Complex::new(-1.0, 2.0));
    }

    #[test]
    fn test_scale() {
        let z = Complex::new(1.0, 2.0);
        assert_eq!(z.scale(2.0), Complex::new(2.0, 4.0));
    }
}
