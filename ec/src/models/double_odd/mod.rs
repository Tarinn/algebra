mod affine;
pub use affine::*;

mod group;
pub use group::*;

use ark_ff::{Zero, Field, LegendreSymbol};
use crate::{AffineRepr, Group};



/// Constants and convenience functions that collectively define the [Double-Odd curve](https://doubleodd.group).
/// In this model, the curve equation is `y² = x(x² + ax + b)`, (b and (a² - 4b) not squares in field)
/// for constants `a` and `b`.!!
pub trait DOCurveConfig: super::CurveConfig {
    /// Coefficient `a` of the curve equation.
    const COEFF_A: Self::BaseField;
    /// Coefficient `b` of the curve equation.
    const COEFF_B: Self::BaseField;
    /// Generator of the prime-order subgroup.
    const GENERATOR: Affine<Self>;

    
    /// Check if the provided curve point is in the prime-order subgroup.
    fn is_in_correct_subgroup_assuming_on_curve(item: &Affine<Self>) -> bool {
        if item.x.is_zero() {
            true
        } else {
            item.x.legendre() == LegendreSymbol::QuadraticNonResidue
        }
    }

    /// Default implementation of group multiplication for projective
    /// coordinates
    fn mul_projective(base: &Projective<Self>, scalar: &[u64]) -> Projective<Self> {
        let mut res = Projective::<Self>::zero();
        for b in ark_ff::BitIteratorBE::without_leading_zeros(scalar) {
            res.double_in_place();
            if b {
                res += base;
            }
        }

        res
    }

    /// Default implementation of group multiplication for affine
    /// coordinates.
    fn mul_affine(base: &Affine<Self>, scalar: &[u64]) -> Projective<Self> {
        let mut res = Projective::<Self>::zero();
        for b in ark_ff::BitIteratorBE::without_leading_zeros(scalar) {
            res.double_in_place();
            if b {
                res += base
            }
        }

        res
    }

    /// If uncompressed, serializes both x and y coordinates as well as a bit for whether it is
    /// infinity. If compressed, serializes x coordinate with two bits to encode whether y is
    /// positive, negative, or infinity.
    #[inline]
    fn serialize_with_mode<W: Write>(
        item: &Affine<Self>,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        let (x, y, flags) = match item.infinity {
            true => (
                Self::BaseField::zero(),
                Self::BaseField::zero(),
                SWFlags::infinity(),
            ),
            false => (item.x, item.y, item.to_flags()),
        };

        match compress {
            Compress::Yes => x.serialize_with_flags(writer, flags),
            Compress::No => {
                x.serialize_with_mode(&mut writer, compress)?;
                y.serialize_with_flags(&mut writer, flags)
            },
        }
    }

    /// If `validate` is `Yes`, calls `check()` to make sure the element is valid.
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Affine<Self>, SerializationError> {
        let (x, y, flags) = match compress {
            Compress::Yes => {
                let (x, flags): (_, SWFlags) =
                    CanonicalDeserializeWithFlags::deserialize_with_flags(reader)?;
                match flags {
                    SWFlags::PointAtInfinity => (
                        Affine::<Self>::identity().x,
                        Affine::<Self>::identity().y,
                        flags,
                    ),
                    _ => {
                        let is_positive = flags.is_positive().unwrap();
                        let (y, neg_y) = Affine::<Self>::get_ys_from_x_unchecked(x)
                            .ok_or(SerializationError::InvalidData)?;
                        if is_positive {
                            (x, y, flags)
                        } else {
                            (x, neg_y, flags)
                        }
                    },
                }
            },
            Compress::No => {
                let x: Self::BaseField =
                    CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
                let (y, flags): (_, SWFlags) =
                    CanonicalDeserializeWithFlags::deserialize_with_flags(&mut reader)?;
                (x, y, flags)
            },
        };
        if flags.is_infinity() {
            Ok(Affine::<Self>::identity())
        } else {
            let point = Affine::<Self>::new_unchecked(x, y);
            if let Validate::Yes = validate {
                point.check()?;
            }
            Ok(point)
        }
    }

    #[inline]
    fn serialized_size(compress: Compress) -> usize {
        let zero = Self::BaseField::zero();
        match compress {
            Compress::Yes => zero.serialized_size_with_flags::<SWFlags>(),
            Compress::No => zero.compressed_size() + zero.serialized_size_with_flags::<SWFlags>(),
        }
    }
}