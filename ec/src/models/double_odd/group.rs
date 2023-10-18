use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    io::{Read, Write},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    vec::Vec,
    One, Zero,
};

use ark_ff::{fields::Field, PrimeField, ToConstraintField, UniformRand};

use zeroize::Zeroize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{Affine, DOCurveConfig};
use crate::{
    scalar_mul::{variable_base::VariableBaseMSM, ScalarMul},
    AffineRepr, CurveGroup, Group,
};

/// Fractional coordinates for a point on an elliptic curve in double odd
/// form, over the base field `P::BaseField`.
#[derive(Derivative)]
#[derivative(Copy(bound = "P: DOCurveConfig"), Clone(bound = "P: DOCurveConfig"), Eq(bound = "P: DOCurveConfig"),)]
#[must_use]
pub struct Projective<P: DOCurveConfig> {
    pub e: P::BaseField,
    pub z: P::BaseField,
    pub u: P::BaseField,
    pub t: P::BaseField,
}

impl<P: DOCurveConfig> Display for Projective<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", Affine::from(*self))
    }
}

impl<P: DOCurveConfig> Debug for Projective<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self.is_zero() {
            true => write!(f, "infinity"),
            false => write!(f, "({}, {}, {}, {})", self.e, self.z, self.u, self.t),
        }
    }
}

impl<P: DOCurveConfig> PartialEq for Projective<P> {
    fn eq(&self, other: &Self) -> bool {
        if self.is_zero() {
            return other.is_zero();
        }

        if other.is_zero() {
            return false;
        }
        // e1/z1 == e2/z2 && u1/t1 == u2/t2 <==> e1*z2 == e2*z1 && u1*t2 == u2*t2
        ((self.e * other.z) == (other.e * self.z)) && ((self.u * other.t) == (other.u * self.t))
    }
}

impl<P: DOCurveConfig> PartialEq<Affine<P>> for Projective<P> {
    fn eq(&self, other: &Affine<P>) -> bool {
        *self == other.into_group()
    }
}

impl<P: DOCurveConfig> Hash for Projective<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.into_affine().hash(state)
    }
}

impl<P: DOCurveConfig> Distribution<Projective<P>> for Standard {
    /// Generates a uniformly random instance of the curve.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Projective<P> {
        loop {
            let x = P::BaseField::rand(rng);
            let greatest = rng.gen();

            if let Some(p) = Affine::get_point_from_x_unchecked(x, greatest) {
                if p.is_in_correct_subgroup_assuming_on_curve() {
                    return p.into()
                } else {
                    return (p + Affine::n()).into()
                }
            }
        }
    }
}

impl<P: DOCurveConfig> Default for Projective<P> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<P: DOCurveConfig> Projective<P> {
    /// Constructs a new group element without checking whether the coordinates
    /// specify a point in the subgroup.
    pub const fn new_unchecked(e: P::BaseField, z: P::BaseField, u: P::BaseField, t: P::BaseField) -> Self {
        Self { e, z, u, t }
    }

    /// Constructs a new group element in a way while enforcing that points are in
    /// the prime-order subgroup.
    pub fn new(e: P::BaseField, z: P::BaseField, u: P::BaseField, t: P::BaseField) -> Self {
        let p = Self::new_unchecked(e, z, u, t).into_affine();
        assert!(p.is_on_curve());
        assert!(p.is_in_correct_subgroup_assuming_on_curve());
        p.into()
    }
}

impl<P: DOCurveConfig> Zeroize for Projective<P> {
    fn zeroize(&mut self) {
        self.e.zeroize();
        self.u.zeroize();
        self.z.zeroize();
        self.t.zeroize();
    }
}

impl<P: DOCurveConfig> Zero for Projective<P> {
    #[inline]
    fn zero() -> Self {
        Self::new_unchecked(
            P::BaseField::zero(),
            P::BaseField::one(),
            P::BaseField::zero(),
            P::BaseField::one(),
        )
    }

    #[inline]
    fn is_zero(&self) -> bool {
        // TODO
        true
    }
}

impl<P: DOCurveConfig> Group for Projective<P> {
    type ScalarField = P::ScalarField;

    #[inline]
    fn generator() -> Self {
        Affine::generator().into()
    }

    fn double_in_place(&mut self) -> &mut Self {
        if self.is_zero() {
            return self;
        }

        // TODO: optimised formula possible depending on P::COEFF_A
        // X' = E^4
        let mut x = self.e; // E
        x.square_in_place().square_in_place(); // E^4

        // W' = 2Z^2 - 2aU - E^2
        let mut w = self.z;
        w.square_in_place().double_in_place(); // 2Z^2
        w = w - (self.u.square().mul(P::COEFF_A)).double() - self.e.square(); // 2Z^2−2aU^2−E^2

        // J'= 2EU
        let mut j = (self.e * self.u).double();
        
        self.z = w.square(); // Z' = W'^2
        self.t = j.square(); // T' = J'^2
        self.u = j * w;      // U' = J' * W'
        self.e = x.double() - self.z + (P::COEFF_A * self.t.double()); // E' = 2X' - Z' + aT'

        self
    }

    #[inline]
    fn mul_bigint(&self, other: impl AsRef<[u64]>) -> Self {
        P::mul_projective(self, other.as_ref())
    }
}

impl<P: DOCurveConfig> CurveGroup for Projective<P> {
    type Config = P;
    type BaseField = P::BaseField;
    type Affine = Affine<P>;
    type FullGroup = Affine<P>;

    #[inline]
    fn normalize_batch(v: &[Self]) -> Vec<Self::Affine> {
        let mut z_s = v.iter().map(|g| g.z).collect::<Vec<_>>();
        ark_ff::batch_inversion(&mut z_s);

        // Perform affine transformations
        ark_std::cfg_iter!(v)
            .zip(z_s)
            .map(|(g, z_i)| match g.is_zero() {
                true => Affine::identity(),
                false => {
                    let e = g.e * z_i;
                    let u = g.u * z_i;

                    let Some(u_2i) = u.square().double().inverse();
                    let x =  u_2i * (e + P::BaseField::ONE - P::COEFF_A * u.square());
                    let y = e;
                    Affine::new_unchecked(x, y)
                },
            })
            .collect()
    }
}

impl<P: DOCurveConfig> Neg for Projective<P> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.u = -self.u;
        self
    }
}

impl<P: DOCurveConfig, T: Borrow<Affine<P>>> AddAssign<T> for Projective<P> {
    fn add_assign(&mut self, other: T) {
        let other = other.borrow();
        let Some(x_i) = other.x.inverse();
        let Some(y_i) = other.y.inverse();
        let u2 = other.x * y_i;
        let e2 = u2.square() * (other.x - P::COEFF_B * x_i);
        let z2 = P::BaseField::ONE;
        let t2 = P::BaseField::ONE;

        let c = P::COEFF_A.square() - P::COEFF_B.double().double();

        let n1 = self.e * e2;
        let n2 = self.z;
        let n3 = self.u * u2;
        let n4 = self.t;
        let n5 = (self.z + self.t) * (z2 + t2) - n2 - n4;
        let n6 = (self.e + self.u) * (e2 + u2) - n1 - n3;
        let n7 = n2 - c * n4;
        
        self.e = (n2 + c * n4)*(n1 - (P::COEFF_A * n2).double()) + (c * n3 * n5).double();
        self.z = n7.square();
        self.t = n6.square();
        self.u = n6 * n7;
    }
}

impl<P: DOCurveConfig, T: Borrow<Affine<P>>> Add<T> for Projective<P> {
    type Output = Self;
    fn add(mut self, other: T) -> Self {
        let other = other.borrow();
        self += other;
        self
    }
}

impl<P: DOCurveConfig, T: Borrow<Affine<P>>> SubAssign<T> for Projective<P> {
    fn sub_assign(&mut self, other: T) {
        *self += -(*other.borrow());
    }
}

impl<P: DOCurveConfig, T: Borrow<Affine<P>>> Sub<T> for Projective<P> {
    type Output = Self;
    fn sub(mut self, other: T) -> Self {
        self -= other.borrow();
        self
    }
}

ark_ff::impl_additive_ops_from_ref!(Projective, DOCurveConfig);

impl<'a, P: DOCurveConfig> Add<&'a Self> for Projective<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &'a Self) -> Self {
        self += other;
        self
    }
}

impl<'a, P: DOCurveConfig> AddAssign<&'a Self> for Projective<P> {
    fn add_assign(&mut self, other: &'a Self) {
        let c = P::COEFF_A.square() - P::COEFF_B.double().double();

        let n1 = self.e * other.e;
        let n2 = self.z * other.z;
        let n3 = self.u * other.u;
        let n4 = self.t * other.t;
        let n5 = (self.z + self.t) * (other.z + other.t) - n2 - n4;
        let n6 = (self.e + self.u) * (other.e + other.u) - n1 - n3;
        let n7 = n2 - c * n4;
        
        self.e = (n2 + c * n4)*(n1 - (P::COEFF_A * n2).double()) + (c * n3 * n5).double();
        self.z = n7.square();
        self.t = n6.square();
        self.u = n6 * n7;
    }
}

impl<'a, P: DOCurveConfig> Sub<&'a Self> for Projective<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &'a Self) -> Self {
        self -= other;
        self
    }
}

impl<'a, P: DOCurveConfig> SubAssign<&'a Self> for Projective<P> {
    fn sub_assign(&mut self, other: &'a Self) {
        *self += &(-(*other));
    }
}

impl<P: DOCurveConfig, T: Borrow<P::ScalarField>> MulAssign<T> for Projective<P> {
    fn mul_assign(&mut self, other: T) {
        *self = self.mul_bigint(other.borrow().into_bigint())
    }
}

impl<P: DOCurveConfig, T: Borrow<P::ScalarField>> Mul<T> for Projective<P> {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: T) -> Self {
        self *= other;
        self
    }
}

impl<P: DOCurveConfig> From<Affine<P>> for Projective<P> {
    #[inline]
    fn from(p: Affine<P>) -> Projective<P> {
        let Some(x_i) = p.x.inverse();
        let Some(y_i) = p.y.inverse();
        let u = p.x * y_i;
        let e = u.square() * (p.x - P::COEFF_B * x_i);
        let z = P::BaseField::ONE;
        let t = P::BaseField::ONE;

        Self::new(e, z, u, t)
    }
}

impl<P: DOCurveConfig> CanonicalSerialize for Projective<P> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let aff = Affine::<P>::from(*self);
        P::serialize_with_mode(&aff, writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        P::serialized_size(compress)
    }
}

impl<P: DOCurveConfig> Valid for Projective<P> {
    fn check(&self) -> Result<(), SerializationError> {
        self.into_affine().check()
    }

    fn batch_check<'a>(
        batch: impl Iterator<Item = &'a Self> + Send,
    ) -> Result<(), SerializationError>
    where
        Self: 'a,
    {
        let batch = batch.copied().collect::<Vec<_>>();
        let batch = Self::normalize_batch(&batch);
        Affine::batch_check(batch.iter())
    }
}

impl<P: DOCurveConfig> CanonicalDeserialize for Projective<P> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let aff = P::deserialize_with_mode(reader, compress, validate)?;
        Ok(aff.into())
    }
}

impl<M: DOCurveConfig, ConstraintF: Field> ToConstraintField<ConstraintF> for Projective<M>
where
    M::BaseField: ToConstraintField<ConstraintF>,
{
    #[inline]
    fn to_field_elements(&self) -> Option<Vec<ConstraintF>> {
        Affine::from(*self).to_field_elements()
    }
}

impl<P: DOCurveConfig> ScalarMul for Projective<P> {
    type MulBase = Affine<P>;

    fn batch_convert_to_mul_base(bases: &[Self]) -> Vec<Self::MulBase> {
        Self::normalize_batch(bases)
    }
}

impl<P: DOCurveConfig> VariableBaseMSM for Projective<P> {}

impl<P: DOCurveConfig, T: Borrow<Affine<P>>> core::iter::Sum<T> for Projective<P> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Projective::zero(), |sum, x| sum + x.borrow())
    }
}

