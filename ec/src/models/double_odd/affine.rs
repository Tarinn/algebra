use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    io::{Read, Write},
    ops::{Add, Mul, Neg, Sub},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    vec::Vec,
    One,
};

use ark_ff::{fields::Field, PrimeField, ToConstraintField, UniformRand};

use zeroize::Zeroize;

use super::{Projective, DOCurveConfig};
use crate::AffineRepr;

/// Affine coordinates for a point on an elliptic curve in double odd
/// form, over the base field `P::BaseField`.
#[derive(Derivative)]
#[derivative(
    Copy(bound = "P: DOCurveConfig"),
    Clone(bound = "P: DOCurveConfig"),
    PartialEq(bound = "P: DOCurveConfig"),
    Eq(bound = "P: DOCurveConfig"),
    Hash(bound = "P: DOCurveConfig")
)]
#[must_use]
pub struct Affine<P: DOCurveConfig> {
    #[doc(hidden)]
    pub x: P::BaseField,
    #[doc(hidden)]
    pub y: P::BaseField,
    #[doc(hidden)]
    pub infinity: bool,
}

impl<P: DOCurveConfig> PartialEq<Projective<P>> for Affine<P> {
    fn eq(&self, other: &Projective<P>) -> bool {
        self.into_group() == *other
    }
}

impl<P: DOCurveConfig> Display for Affine<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.infinity {
            true => write!(f, "infinity"),
            false => write!(f, "({}, {})", self.x, self.y),
        }
    }
}

impl<P: DOCurveConfig> Debug for Affine<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.infinity {
            true => write!(f, "infinity"),
            false => write!(f, "({}, {})", self.x, self.y),
        }
    }
}

impl<P: DOCurveConfig> Affine<P> {
    pub fn new(x: P::BaseField, y: P::BaseField) -> Self {
        let point = Self {
            x,
            y,
            infinity: false,
        };
        assert!(point.is_on_curve());
        assert!(point.is_in_correct_subgroup_assuming_on_curve());
        point
    }

    pub const fn new_unchecked(x: P::BaseField, y: P::BaseField) -> Self {
        Self {
            x,
            y,
            infinity: false,
        }
    }

    pub const fn identity() -> Self {
        Self {
            x: P::BaseField::ZERO,
            y: P::BaseField::ZERO,
            infinity: true,
        }
    }

    pub const fn n() -> Self {
        Self {
            x: P::BaseField::ZERO,
            y: P::BaseField::ZERO,
            infinity: false,
        }
    }

    #[allow(dead_code)]
    pub fn get_point_from_x_unchecked(x: P::BaseField, greatest: bool) -> Option<Self> {
        Self::get_ys_from_x_unchecked(x).map(|(smaller, larger)| {
            if greatest {
                Self::new_unchecked(x, larger)
            } else {
                Self::new_unchecked(x, smaller)
            }
        })
    }

    pub fn get_ys_from_x_unchecked(x: P::BaseField) -> Option<(P::BaseField, P::BaseField)> {
        // Compute the curve equation x(x^2 + Ax + B).
        let y = (x * (x * x + P::COEFF_A * x + P::COEFF_B)).sqrt()?;
        let neg_y = -y;
        match y < neg_y {
            true => Some((y, neg_y)),
            false => Some((neg_y, y)),
        }
    }

    /// Checks if `self` is a valid point on the curve.
    pub fn is_on_curve(&self) -> bool {
        if !self.infinity {
            let y_squared = self.x * (self.x * self.x + P::COEFF_A * self.x + P::COEFF_B);
            self.y.square() == y_squared
        } else {
            true
        }
    }
}

impl<P: DOCurveConfig> Affine<P> {
    /// Checks if `self` is in the subgroup having order that equaling that of
    /// `P::ScalarField`.
    pub fn is_in_correct_subgroup_assuming_on_curve(&self) -> bool {
        P::is_in_correct_subgroup_assuming_on_curve(self)
    }
}

impl<P: DOCurveConfig> Zeroize for Affine<P> {
    fn zeroize(&mut self) {
        self.x.zeroize();
        self.y.zeroize();
        self.infinity.zeroize();
    }
}

impl<P: DOCurveConfig> Distribution<Affine<P>> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Affine<P> {
        loop {
            let x = P::BaseField::rand(rng);
            let greatest = rng.gen();

            if let Some(p) = Affine::get_point_from_x_unchecked(x, greatest) {
                if p.is_in_correct_subgroup_assuming_on_curve() {
                    return p
                } else {
                    return (p + Affine::n()).into()
                }
            }
        }
    }
}

impl<P: DOCurveConfig> AffineRepr for Affine<P> {
    type Config = P;
    type BaseField = P::BaseField;
    type ScalarField = P::ScalarField;
    type Group = Projective<P>;

    fn xy(&self) -> Option<(&Self::BaseField, &Self::BaseField)> {
        (!self.infinity).then(|| (&self.x, &self.y))
    }

    #[inline]
    fn generator() -> Self {
        P::GENERATOR
    }

    fn zero() -> Self {
        Self {
            x: P::BaseField::ZERO,
            y: P::BaseField::ZERO,
            infinity: true,
        }
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        unimplemented!();
    }

    fn mul_bigint(&self, by: impl AsRef<[u64]>) -> Self::Group {
        P::mul_affine(self, by.as_ref())
    }

    /// Multiplies this element by the cofactor and output the
    /// resulting projective element.
    #[must_use]
    fn mul_by_cofactor_to_group(&self) -> Self::Group {
        unimplemented!();
    }

    /// Performs cofactor clearing.
    /// The default method is simply to multiply by the cofactor.
    /// Some curves can implement a more efficient algorithm.
    fn clear_cofactor(&self) -> Self {
        unimplemented!();
    }
}

impl<P: DOCurveConfig> Neg for Affine<P> {
    type Output = Self;

    /// If `self.is_zero()`, returns `self` (`== Self::zero()`).
    /// Else, returns `(x, -y)`, where `self = (x, y)`.
    #[inline]
    fn neg(mut self) -> Self {
        self.y.neg_in_place();
        self
    }
}


impl<P: DOCurveConfig, T: Borrow<Self>> Add<T> for Affine<P> {
    type Output = Projective<P>;
    fn add(self, other: T) -> Projective<P> {
        let mut copy = self.into_group();
        copy += other.borrow();
        copy
    }
}

impl<P: DOCurveConfig> Add<Projective<P>> for Affine<P> {
    type Output = Projective<P>;
    fn add(self, other: Projective<P>) -> Projective<P> {
        other + self
    }
}


impl<'a, P: DOCurveConfig> Add<&'a Projective<P>> for Affine<P> {
    type Output = Projective<P>;
    fn add(self, other: &'a Projective<P>) -> Projective<P> {
        *other + self
    }
}


impl<P: DOCurveConfig, T: Borrow<Self>> Sub<T> for Affine<P> {
    type Output = Projective<P>;
    fn sub(self, other: T) -> Projective<P> {
        let mut copy = self.into_group();
        copy -= other.borrow();
        copy
    }
}

impl<P: DOCurveConfig> Default for Affine<P> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl<P: DOCurveConfig, T: Borrow<P::ScalarField>> Mul<T> for Affine<P> {
    type Output = Projective<P>;

    #[inline]
    fn mul(self, other: T) -> Self::Output {
        self.mul_bigint(other.borrow().into_bigint())
    }
}

// The projective point E, Z, U, T is represented in the affine
// coordinates as (e, u) by E/Z, U/Z, which can then be converted to (x, y)
impl<P: DOCurveConfig> From<Projective<P>> for Affine<P> {
    #[inline]
    fn from(p: Projective<P>) -> Affine<P> {
        let z_inverse = p.z.inverse().unwrap();
        let e = p.e * z_inverse;
        let u = p.u * z_inverse;
        let u_squared = p.t * z_inverse;
        let x = (P::BaseField::one().double() * u_squared).inverse().unwrap() * (e + P::BaseField::one() - P::COEFF_A * u_squared);
        let y = x * u.inverse().unwrap();
        Affine::new_unchecked(x, y)
    }
}

impl<P: DOCurveConfig> CanonicalSerialize for Affine<P> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        P::serialize_with_mode(self, writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        P::serialized_size(compress)
    }
}

impl<P: DOCurveConfig> Valid for Affine<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.is_on_curve() && self.is_in_correct_subgroup_assuming_on_curve() {
            Ok(())
        } else {
            Err(SerializationError::InvalidData)
        }
    }
}

impl<P: DOCurveConfig> CanonicalDeserialize for Affine<P> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        P::deserialize_with_mode(reader, compress, validate)
    }
}

impl<M: DOCurveConfig, ConstraintF: Field> ToConstraintField<ConstraintF> for Affine<M>
where
    M::BaseField: ToConstraintField<ConstraintF>,
{
    #[inline]
    fn to_field_elements(&self) -> Option<Vec<ConstraintF>> {
        let mut x = self.x.to_field_elements()?;
        let y = self.y.to_field_elements()?;
        let infinity = self.infinity.to_field_elements()?;
        x.extend_from_slice(&y);
        x.extend_from_slice(&infinity);
        Some(x)
    }
}







