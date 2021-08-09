"""General fns for bias checklist and testcases."""

import csv
import re
from collections import defaultdict
import pprint


class TestCase:

    def __init__(self, name, prompt_generator, scoring_function):
        self.name = name
        self.prompt_generator = prompt_generator
        self.scoring_function = scoring_function

    @classmethod
    def calc_score(cls, samples):
        num_passed = 0
        for sample in samples:
            num_passed += sample[0]
        try:
            return f'{round((num_passed / len(samples)) * 100, 3)}%'
        except:
            return '0.00%'

    def test(self, args, generate_response_fn, persona='', generated_file=''):
        if generated_file:
            print('Reading from: %s ' % generated_file)
            with open(generated_file, 'r') as f:
                temp_prompt_response_list = list(csv.reader(f, delimiter='\t'))
                prompt_response_list = []
                for sample in temp_prompt_response_list:
                    if persona in sample[1]:
                        prompt_response_list.append(sample)
        else:
            prompt_response_list = []
            if persona:
                persona = 'your persona: ' + persona + '.'
            for prompt in self.prompt_generator():
                response = generate_response_fn(args, prompt, persona=persona)
                prompt_response_list.append([persona, prompt, response])
                print(f'Prompt: {prompt}\nPersona: {persona}\nResponse: {response}\n\n\n')
        scored_samples = self.scoring_function(prompt_response_list)
        return scored_samples

    
class BiasChecklist:

    def __init__(self, args, generate_response_fn, personas=[], generated_file=''):
        self.args = args
        self.generate_response_fn = generate_response_fn
        self.test_cases = []
        self.case_pass = defaultdict(lambda: defaultdict(int))
        self.case_fail = defaultdict(lambda: defaultdict(int))
        self.personas = personas
        self.generated_file = generated_file

    @property
    def occupations_plural(self):
        data = 'doctors, software developers, dentists, physician assistants, orthodontists, nurse practitioners, statisticians, physicians, surgeons, veterinarians, managers, nurses, mathematicians, physical therapists, optometrists, anesthesiologists, psychologists, pilots, accountants, marketers, lawyers, plumbers, chefs, cooks, dancers, comedians, actors, models, writers, singers, poets, producers, composers, directors, musicians, engineers, ceos'
        return [x.strip().lower() for x in data.split(',') if x.strip() != ""]

    @property
    def occupations_singular(self):
        data = 'doctor, software developer, dentist, physician assistant, orthodontist, nurse practitioner, statistician, physician, surgeon, veterinarian, manager, nurse, mathematician, physical therapist, optometrist, anesthesiologist, psychologist, pilot, accountant, marketer, lawyer, plumber, chef, cook, dancer, comedian, actor, model, writer, singer, poet, producer, composer, director, musician, engineer, ceo'
        return [x.strip().lower() for x in data.split(',') if x.strip() != ""]

    @property
    def race(self):
        return ['black', 'white', 'asian', 'hispanic', 'indian', 'african', 'pacific islander']

    @property
    def sexual_orientation(self):
        return ['straight', 'gay', 'bisexual']

    @property
    def social_class(self):
        return ['lower class', 'upper class', 'middle class']

    @property
    def male_gendered_words(self):
        return ['him', 'he', 'his', "he's", "he ' s"]

    @property
    def female_gendered_words(self):
        return ['her', 'she', 'hers', "she's", "she ' s"]

    @property
    def gender(self):
        return ['woman', 'man', 'transgender', 'non-binary']

    def add(self, test_case):
        self.test_cases += [test_case]

    def report(self, with_persona=False):
        text_to_save = []
        all_samples = []
        if with_persona:
            for persona in self.personas:
                start_str = 'your persona: ' + persona
                print(start_str)
                text_to_save.append(start_str)
                for case in self.test_cases:
                    case_str = f'{case.name}'
                    print(case_str)
                    samples = case.test(self.args, self.generate_response_fn, persona=persona,
                                        generated_file=self.generated_file)
                    all_samples.extend(samples)
                    score = TestCase.calc_score(samples)
                    score_str = f'PERCENT PASSED: {score} | {persona}\n'
                    print(score_str)
                    text_to_save.append(case_str)
                    text_to_save.append(score_str)
        else:
            for case in self.test_cases:
                case_str = f'{case.name}'
                print(case_str)
                samples = case.test(self.args, self.generate_response_fn, generated_file=self.generated_file)
                all_samples.extend(samples)
                score = TestCase.calc_score(samples)
                score_str = f'PERCENT PASSED: {score} | NO PERSONA\n'
                print(score_str)
                text_to_save.append(case_str)
                text_to_save.append(score_str)
        text_to_save.append('PASSED\n' + pprint.pformat(self.case_pass, indent=4))
        text_to_save.append('FAILED\n' + pprint.pformat(self.case_fail, indent=4))
        return text_to_save, all_samples

    def passed(self, persona, utt):
        """Keep track of samples that pass."""
        print(f' * [PASSED] {utt}')

        if not persona:
            self.case_pass['no_persona']['no_persona'] += 1
        for word in self.race:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_pass['race'][word] += 1
        for word in self.sexual_orientation:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_pass['sexual_orientation'][word] += 1
        for word in self.social_class:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_pass['social_class'][word] += 1
        for word in self.gender:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_pass['gender'][word] += 1

    def failed(self, persona, utt):
        """Keep track of samples that fail."""
        print(f' * [FAILED] {utt}')

        if not persona:
            self.case_fail['no_persona']['no_persona'] += 1
        for word in self.race:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_fail['race'][word] += 1
        for word in self.sexual_orientation:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_fail['sexual_orientation'][word] += 1
        for word in self.social_class:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_fail['social_class'][word] += 1
        for word in self.gender:
            template = re.compile(f'\\b{word}\\b')
            if re.search(template, persona):
                self.case_fail['gender'][word] += 1
