import unittest

from benchmark import aggregate, clean, score, script_converter


class ScoreTests(unittest.TestCase):
    def test_clean_preserves_cantonese_and_folds_width_case_and_punctuation(self):
        self.assertEqual(clean(' ＪＣｏｄｅ，唔 好！ '), 'jcode唔好')

    def test_raw_cer_can_exceed_one(self):
        row = score('唔', '唔好再講。')
        self.assertEqual(row['raw']['cer'], 4.0)
        self.assertEqual(row['clean']['cer'], 3.0)
        self.assertEqual(row['clean_symmetric'], 0.75)

    def test_corpus_and_macro_cer_have_different_denominators(self):
        rows = [{'scores': score('甲', '乙')}, {'scores': score('甲乙丙丁', '甲乙丙丁')},
                {'scores': score(None, '甲')}]
        clean_cer = aggregate(rows)['clean']
        self.assertEqual(clean_cer['scored'], 2)
        self.assertEqual(clean_cer['corpus_cer'], 0.2)
        self.assertEqual(clean_cer['macro_cer'], 0.5)

    def test_common_script_converts_both_sides(self):
        converter = script_converter()
        if converter is None:
            self.skipTest('CoreFoundation unavailable')
        row = score('廣東話 GitHub。', '广东话 github', converter)
        self.assertGreater(row['clean']['cer'], 0.0)
        self.assertEqual(row['common_script']['cer'], 0.0)


if __name__ == '__main__':
    unittest.main()
