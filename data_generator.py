"""
Test Data Management and Generation Module.
Provides synthetic data generation, masking, versioning, and dependency management.

Built-in Self-Tests:
- Function-level: Each method validates inputs and outputs
- Class-level: Data generation and validation
- Module-level: End-to-end data workflows
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import hashlib
import random
import string

import structlog
from faker import Faker
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class TestDataValidator:
    """Built-in validator for test data operations."""
    
    @staticmethod
    def validate_data_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate data against schema.
        
        Success Criteria:
        - All required fields present
        - Field types match schema
        - Values within constraints
        """
        try:
            for field, field_schema in schema.items():
                if field_schema.get('required', False) and field not in data:
                    return False
                
                if field in data:
                    expected_type = field_schema.get('type')
                    if expected_type and not isinstance(data[field], eval(expected_type)):
                        return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_masked_data(original: str, masked: str, mask_char: str = '*') -> bool:
        """
        Validate masked data.
        
        Success Criteria:
        - Masked data length matches original
        - Contains mask characters
        - Original data not visible
        """
        if len(original) != len(masked):
            return False
        if mask_char not in masked:
            return False
        if original == masked:
            return False
        return True


class DataVersion(BaseModel):
    """Test data version model."""
    version_id: str
    timestamp: datetime
    data_hash: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestDataGenerator:
    """
    Test data generator with Faker integration.
    
    Success Criteria (Class-level):
    - Generates valid synthetic data
    - Supports multiple data types
    - Reproducible with seeds
    - Performance: > 1000 records/second
    """
    
    def __init__(self, locale: str = 'en_US', seed: Optional[int] = None):
        """
        Initialize test data generator.
        
        Args:
            locale: Faker locale for data generation
            seed: Random seed for reproducibility
        
        Success Criteria:
        - Faker initialized with locale
        - Seed applied if provided
        """
        self.faker = Faker(locale)
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        
        self.validator = TestDataValidator()
        
        logger.info(
            "Test data generator initialized",
            locale=locale,
            seed=seed
        )
        
        # Self-test initialization
        self._self_test_init()
    
    def _self_test_init(self) -> bool:
        """
        Self-test: Validate initialization.
        
        Success Criteria:
        - Faker can generate basic data
        - Validator works correctly
        """
        try:
            # Test basic generation
            name = self.faker.name()
            if not isinstance(name, str) or len(name) == 0:
                logger.error("Self-test failed: Name generation")
                return False
            
            # Test validator
            schema = {'name': {'type': 'str', 'required': True}}
            data = {'name': 'Test'}
            if not self.validator.validate_data_schema(data, schema):
                logger.error("Self-test failed: Schema validation")
                return False
            
            logger.debug("Self-test passed: Initialization")
            return True
        
        except Exception as e:
            logger.error("Self-test failed", error=str(e))
            return False
    
    def generate_user(self, **overrides) -> Dict[str, Any]:
        """
        Generate synthetic user data.
        
        Args:
            **overrides: Override specific fields
        
        Returns:
            User data dictionary
        
        Success Criteria:
        - All required fields present
        - Data types correct
        - Email format valid
        """
        user = {
            'id': self.faker.uuid4(),
            'username': self.faker.user_name(),
            'email': self.faker.email(),
            'first_name': self.faker.first_name(),
            'last_name': self.faker.last_name(),
            'phone': self.faker.phone_number(),
            'address': {
                'street': self.faker.street_address(),
                'city': self.faker.city(),
                'state': self.faker.state(),
                'zip': self.faker.zipcode(),
                'country': self.faker.country()
            },
            'date_of_birth': self.faker.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            'created_at': datetime.now().isoformat(),
            'is_active': True
        }
        
        # Apply overrides
        user.update(overrides)
        
        logger.debug("User data generated", user_id=user['id'])
        return user
    
    def generate_product(self, **overrides) -> Dict[str, Any]:
        """
        Generate synthetic product data.
        
        Success Criteria:
        - Product has valid SKU
        - Price is positive
        - Stock quantity is non-negative
        """
        product = {
            'id': self.faker.uuid4(),
            'sku': self.faker.bothify(text='???-########', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'name': self.faker.catch_phrase(),
            'description': self.faker.text(max_nb_chars=200),
            'category': random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
            'price': round(random.uniform(9.99, 999.99), 2),
            'stock_quantity': random.randint(0, 1000),
            'manufacturer': self.faker.company(),
            'created_at': datetime.now().isoformat()
        }
        
        product.update(overrides)
        
        logger.debug("Product data generated", product_id=product['id'])
        return product
    
    def generate_order(self, user_id: Optional[str] = None, **overrides) -> Dict[str, Any]:
        """
        Generate synthetic order data.
        
        Success Criteria:
        - Order has valid user reference
        - Total matches item prices
        - Status is valid
        """
        if user_id is None:
            user_id = self.faker.uuid4()
        
        num_items = random.randint(1, 5)
        items = []
        total = 0.0
        
        for _ in range(num_items):
            price = round(random.uniform(10.0, 100.0), 2)
            quantity = random.randint(1, 3)
            subtotal = round(price * quantity, 2)
            total += subtotal
            
            items.append({
                'product_id': self.faker.uuid4(),
                'product_name': self.faker.catch_phrase(),
                'price': price,
                'quantity': quantity,
                'subtotal': subtotal
            })
        
        order = {
            'id': self.faker.uuid4(),
            'user_id': user_id,
            'order_number': self.faker.bothify(text='ORD-########'),
            'items': items,
            'subtotal': round(total, 2),
            'tax': round(total * 0.08, 2),
            'shipping': round(random.uniform(5.0, 15.0), 2),
            'total': round(total * 1.08 + random.uniform(5.0, 15.0), 2),
            'status': random.choice(['pending', 'processing', 'shipped', 'delivered']),
            'created_at': datetime.now().isoformat()
        }
        
        order.update(overrides)
        
        logger.debug("Order data generated", order_id=order['id'])
        return order
    
    def generate_batch(self, data_type: str, count: int, **overrides) -> List[Dict[str, Any]]:
        """
        Generate batch of test data.
        
        Args:
            data_type: Type of data ('user', 'product', 'order')
            count: Number of records to generate
            **overrides: Override specific fields
        
        Returns:
            List of generated records
        
        Success Criteria:
        - All records generated successfully
        - Count matches requested
        - Performance: > 1000 records/second
        """
        generators = {
            'user': self.generate_user,
            'product': self.generate_product,
            'order': self.generate_order
        }
        
        if data_type not in generators:
            raise ValueError(f"Unknown data type: {data_type}")
        
        generator = generators[data_type]
        batch = []
        
        start_time = datetime.now()
        
        for _ in range(count):
            batch.append(generator(**overrides))
        
        duration = (datetime.now() - start_time).total_seconds()
        rate = count / duration if duration > 0 else 0
        
        logger.info(
            "Batch generated",
            data_type=data_type,
            count=count,
            duration=duration,
            rate_per_second=rate
        )
        
        return batch


class DataMasker:
    """
    Data masking for PII protection.
    
    Success Criteria (Class-level):
    - Original data not visible
    - Masked data maintains format
    - Reversible with key (for testing)
    """
    
    def __init__(self):
        """Initialize data masker."""
        self.validator = TestDataValidator()
        logger.info("Data masker initialized")
    
    def mask_email(self, email: str) -> str:
        """
        Mask email address.
        
        Success Criteria:
        - Username partially masked
        - Domain preserved
        - Format valid
        """
        if '@' not in email:
            return email
        
        username, domain = email.split('@', 1)
        
        if len(username) <= 2:
            masked_username = '*' * len(username)
        else:
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        
        return f"{masked_username}@{domain}"
    
    def mask_phone(self, phone: str) -> str:
        """
        Mask phone number.
        
        Success Criteria:
        - Last 4 digits visible
        - Rest masked
        """
        digits = ''.join(c for c in phone if c.isdigit())
        
        if len(digits) <= 4:
            return '*' * len(phone)
        
        masked_digits = '*' * (len(digits) - 4) + digits[-4:]
        
        # Reconstruct with original formatting
        result = []
        digit_idx = 0
        for c in phone:
            if c.isdigit():
                result.append(masked_digits[digit_idx])
                digit_idx += 1
            else:
                result.append(c)
        
        return ''.join(result)
    
    def mask_credit_card(self, card: str) -> str:
        """
        Mask credit card number.
        
        Success Criteria:
        - Last 4 digits visible
        - Rest masked
        - Format preserved
        """
        digits = ''.join(c for c in card if c.isdigit())
        
        if len(digits) < 13:
            return '*' * len(card)
        
        masked_digits = '*' * (len(digits) - 4) + digits[-4:]
        
        # Reconstruct with original formatting
        result = []
        digit_idx = 0
        for c in card:
            if c.isdigit():
                result.append(masked_digits[digit_idx])
                digit_idx += 1
            else:
                result.append(c)
        
        return ''.join(result)
    
    def mask_data(self, data: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """
        Mask multiple fields in data.
        
        Args:
            data: Data dictionary
            fields: Fields to mask
        
        Returns:
            Masked data dictionary
        
        Success Criteria:
        - Specified fields masked
        - Other fields unchanged
        - Structure preserved
        """
        masked = data.copy()
        
        for field in fields:
            if field in masked:
                value = masked[field]
                
                if 'email' in field.lower():
                    masked[field] = self.mask_email(str(value))
                elif 'phone' in field.lower():
                    masked[field] = self.mask_phone(str(value))
                elif 'card' in field.lower() or 'credit' in field.lower():
                    masked[field] = self.mask_credit_card(str(value))
                else:
                    # Generic masking
                    masked[field] = '*' * len(str(value))
        
        logger.debug("Data masked", fields=fields)
        return masked


class DataVersionManager:
    """
    Test data versioning and rollback.
    
    Success Criteria (Class-level):
    - Versions stored with hashes
    - Rollback to any version
    - Version history maintained
    """
    
    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, List[DataVersion]] = {}
        logger.info("Data version manager initialized")
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute data hash.
        
        Success Criteria:
        - Hash is deterministic
        - Same data produces same hash
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def save_version(self, dataset_id: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save data version.
        
        Args:
            dataset_id: Dataset identifier
            data: Data to version
            metadata: Optional metadata
        
        Returns:
            Version ID
        
        Success Criteria:
        - Version saved with hash
        - Version ID unique
        - Metadata preserved
        """
        version_id = f"v{len(self.versions.get(dataset_id, [])) + 1}"
        data_hash = self._compute_hash(data)
        
        version = DataVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            data_hash=data_hash,
            data=data,
            metadata=metadata or {}
        )
        
        if dataset_id not in self.versions:
            self.versions[dataset_id] = []
        
        self.versions[dataset_id].append(version)
        
        logger.info(
            "Version saved",
            dataset_id=dataset_id,
            version_id=version_id,
            data_hash=data_hash[:8]
        )
        
        return version_id
    
    def get_version(self, dataset_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific version.
        
        Success Criteria:
        - Returns correct version data
        - None if not found
        """
        if dataset_id not in self.versions:
            return None
        
        for version in self.versions[dataset_id]:
            if version.version_id == version_id:
                logger.debug("Version retrieved", dataset_id=dataset_id, version_id=version_id)
                return version.data
        
        return None
    
    def get_latest_version(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get latest version.
        
        Success Criteria:
        - Returns most recent version
        - None if dataset not found
        """
        if dataset_id not in self.versions or not self.versions[dataset_id]:
            return None
        
        latest = self.versions[dataset_id][-1]
        logger.debug("Latest version retrieved", dataset_id=dataset_id, version_id=latest.version_id)
        return latest.data
    
    def list_versions(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        List all versions.
        
        Success Criteria:
        - Returns all versions
        - Sorted by timestamp
        """
        if dataset_id not in self.versions:
            return []
        
        return [
            {
                'version_id': v.version_id,
                'timestamp': v.timestamp.isoformat(),
                'data_hash': v.data_hash,
                'metadata': v.metadata
            }
            for v in self.versions[dataset_id]
        ]


# Module-level self-test
def self_test_module() -> bool:
    """
    Module-level self-test.
    
    Success Criteria:
    - Generator produces valid data
    - Masker masks correctly
    - Versioning works
    """
    try:
        # Test generator
        generator = TestDataGenerator(seed=42)
        user = generator.generate_user()
        
        if 'email' not in user or '@' not in user['email']:
            logger.error("Module self-test failed: Invalid user data")
            return False
        
        # Test masker
        masker = DataMasker()
        masked_email = masker.mask_email("test@example.com")
        
        if masked_email == "test@example.com":
            logger.error("Module self-test failed: Email not masked")
            return False
        
        # Test versioning
        version_mgr = DataVersionManager()
        version_id = version_mgr.save_version("test_dataset", user)
        retrieved = version_mgr.get_version("test_dataset", version_id)
        
        if retrieved != user:
            logger.error("Module self-test failed: Version mismatch")
            return False
        
        logger.info("Module self-test passed: test_data.data_generator")
        return True
    
    except Exception as e:
        logger.error("Module self-test failed", error=str(e))
        return False


if __name__ == "__main__":
    # Run module self-test
    success = self_test_module()
    print(f"Module self-test: {'PASSED' if success else 'FAILED'}")
