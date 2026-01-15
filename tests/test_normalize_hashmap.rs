use gemini_structured_output::schema::normalize_json_response;
use serde_json::json;

#[test]
fn test_normalize_account_overrides() {
    let mut response = json!({
        "pnlConfig": {
            "forecastConfig": {
                "accountOverrides": [
                    {
                        "__key__": "Access Advisors - Sales",
                        "__value__": {
                            "processor": {
                                "modelType": "mstl",
                                "seasonalPeriods": [12],
                                "trendModel": "ets"
                            }
                        }
                    },
                    {
                        "__key__": "Tax Provision",
                        "__value__": {
                            "processor": {
                                "taxRate": 0.28
                            }
                        }
                    }
                ]
            }
        }
    });

    println!("Before normalization:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    normalize_json_response(&mut response);

    println!("\nAfter normalization:");
    println!("{}", serde_json::to_string_pretty(&response).unwrap());

    // Verify accountOverrides was converted from array to object
    let account_overrides = response
        .get("pnlConfig")
        .and_then(|p| p.get("forecastConfig"))
        .and_then(|f| f.get("accountOverrides"))
        .expect("accountOverrides should exist");

    assert!(
        account_overrides.is_object(),
        "accountOverrides should be an object after normalization, but got: {}",
        account_overrides
    );

    let obj = account_overrides.as_object().unwrap();
    assert!(
        obj.contains_key("Access Advisors - Sales"),
        "Should have key 'Access Advisors - Sales'"
    );
    assert!(
        obj.contains_key("Tax Provision"),
        "Should have key 'Tax Provision'"
    );

    println!("✅ normalize_json_response correctly converted accountOverrides from array to object");
}
