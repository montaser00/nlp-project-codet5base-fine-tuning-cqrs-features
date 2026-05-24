

    def test_find_by_id(self):
        self.assertEqual(
            self._get_query(id=1).count(), 1)
