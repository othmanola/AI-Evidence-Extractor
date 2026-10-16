import unittest

import update_apply
import update_client


class UpdateMigrationTests(unittest.TestCase):
    def test_account_module_is_allowed_but_not_required_for_compatibility_updates(self):
        self.assertIn("account_client.py", update_client.UPDATE_FILES)
        self.assertIn("account_client.py", update_client.OPTIONAL_UPDATE_FILES)
        self.assertIn("account_client.py", update_apply.ALLOWED)
        self.assertNotIn("account_client.py", update_apply.REQUIRED)


if __name__ == "__main__":
    unittest.main()
