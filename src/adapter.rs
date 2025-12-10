use std::hash::Hash;
use std::time::Duration;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use schemars::JsonSchema;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// A helper struct that represents a map entry as a structured object.
/// This acts as the "wire format" for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct KeyValue<K, V> {
    pub key: K,
    pub value: V,
}

/// Serde adapter module for generic maps (e.g., `HashMap`, `BTreeMap`).
///
/// # Usage
/// ```rust,ignore
/// #[serde(with = "gemini_structured_output::adapter::map")]
/// #[schemars(with = "Vec<gemini_structured_output::adapter::KeyValue<String, f64>>")]
/// pub my_map: HashMap<String, f64>;
/// ```
pub mod map {
    use super::*;

    pub fn serialize<K, V, S, M>(map: &M, serializer: S) -> Result<S::Ok, S::Error>
    where
        for<'a> &'a M: IntoIterator<Item = (&'a K, &'a V)>,
        K: Serialize + Clone,
        V: Serialize + Clone,
        S: Serializer,
    {
        let entries: Vec<KeyValue<K, V>> = <&M as IntoIterator>::into_iter(map)
            .map(|(key, value)| KeyValue {
                key: key.clone(),
                value: value.clone(),
            })
            .collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, K, V, D, M>(deserializer: D) -> Result<M, D::Error>
    where
        K: Deserialize<'de> + Hash + Eq + Ord,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
        M: FromIterator<(K, V)>,
    {
        let entries: Vec<KeyValue<K, V>> = Vec::deserialize(deserializer)?;
        Ok(entries.into_iter().map(|kv| (kv.key, kv.value)).collect())
    }
}

/// Serialize Vec<T> as a map keyed by UUIDs to make patching order-independent.
pub mod robust_vec {
    use super::*;

    pub fn serialize<T, S>(vec: &Vec<T>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Serialize,
        S: Serializer,
    {
        let mut map = HashMap::new();
        for item in vec {
            map.insert(Uuid::new_v4().to_string(), item);
        }
        map.serialize(serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
    where
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let map: HashMap<String, T> = HashMap::deserialize(deserializer)?;
        Ok(map.into_values().collect())
    }
}

/// Serialize HashMap<K, V> as Vec<KeyValue<K, V>> to guide LLMs with explicit key/value pairs.
pub mod kv_map {
    use super::*;

    pub fn serialize<K, V, S>(map: &HashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: Serialize + Clone + Eq + Hash,
        V: Serialize + Clone,
        S: Serializer,
    {
        let entries: Vec<KeyValue<K, V>> = map
            .iter()
            .map(|(k, v)| KeyValue {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let entries: Vec<KeyValue<K, V>> = Vec::deserialize(deserializer)?;
        let mut map = HashMap::new();
        for entry in entries {
            map.insert(entry.key, entry.value);
        }
        Ok(map)
    }
}

/// Serializes `Vec<u8>` as a Base64 string instead of an integer array to keep payloads compact.
pub mod base64_bytes {
    use super::*;

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&BASE64.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        BASE64.decode(s).map_err(serde::de::Error::custom)
    }
}

/// Serializes `Duration` as simple seconds (f64). Easier for the LLM to produce accurately.
pub mod duration_secs {
    use super::*;

    pub fn serialize<S>(dur: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(dur.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

/// Serializes numbers as strings and accepts either string or integer on input.
/// Useful when LLMs prefer to emit numeric strings.
pub mod string_or_int {
    use super::*;

    pub fn serialize<S>(val: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&val.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StringOrIntVisitor;

        impl<'de> Visitor<'de> for StringOrIntVisitor {
            type Value = u64;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("string or integer")
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(value)
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse::<u64>().map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_any(StringOrIntVisitor)
    }
}
