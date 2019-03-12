#!/usr/bin/env php
<?php declare(strict_types=1);
namespace dliPHPML;
ini_set("memory_limit", "-1");
include 'vendor/autoload.php';

use Phpml\Dataset\ArrayDataset;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\Pipeline;
use Phpml\Tokenization\WordTokenizer;
use Phpml\CrossValidation\StratifiedRandomSplit;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\Metric\Accuracy;
use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;
use Stichoza\GoogleTranslate\GoogleTranslate;

/*
 * - If sentences.txt exists
 *      - If languagedataset.ser does not exist
 *          - Setup inital english sentences from sentences.txt
 * - If sentences.txt contains new sentences or have removed sentences
 *      - Update english sentences in dataset
 * - For each language
 *      - Check if english sentence exist that is missing for language
 *          - Translate each missing sentence using Google Translate
 *      - Store updated languagedataset.ser
 *
 * - If model.dat already exists or we have to retrain due to changed dataset
 *      - Transform format from languagedataset.ser to an ArrayDataset, train and check accurcy
 *      - Save model.dat
 * - Else load mode.dat
 *      - Predict language of sentences passed
 */
 
 // Number of sentences to use for each language when training
 numSentences = 400;

srand();

// Keeps track of if we should retrain an existing model. If dataset has been changed the model will retrain
$train = false;

$languages = [
    'af' => 'Afrikaans',
    'sq' => 'Albanian',
    'ar' => 'Arabic',
    'az' => 'Azerbaijani',
    'eu' => 'Basque',
    'bn' => 'Bengali',
    'be' => 'Belarusian',
    'bg' => 'Bulgarian',
    'ca' => 'Catalan',
    'zh-CN' => 'Chinese Simplified',
    'zh-TW' => 'Chinese Traditional',
    'hr' => 'Croatian',
    'cs' => 'Czech',
    'da' => 'Danish',
    'nl' => 'Dutch',
    'en' => 'English',
    'eo' => 'Esperanto',
    'et' => 'Estonian',
    'tl' => 'Filipino',
    'fi' => 'Finnish',
    'fr' => 'French',
    'gl' => 'Galician',
    'ka' => 'Georgian',
    'de' => 'German',
    'el' => 'Greek',
    'gu' => 'Gujarati',
    'ht' => 'Haitian Creole',
    'iw' => 'Hebrew',
    'hi' => 'Hindi',
    'hu' => 'Hungarian',
    'is' => 'Icelandic',
    'id' => 'Indonesian',
    'ga' => 'Irish',
    'it' => 'Italian',
    'ja' => 'Japanese',
    'kn' => 'Kannada',
    'ko' => 'Korean',
    'la' => 'Latin',
    'lv' => 'Latvian',
    'lt' => 'Lithuanian',
    'mk' => 'Macedonian',
    'ms' => 'Malay',
    'mt' => 'Maltese',
    'no' => 'Norwegian',
    'fa' => 'Persian',
    'pl' => 'Polish',
    'pt' => 'Portuguese',
    'ro' => 'Romanian',
    'ru' => 'Russian',
    'sr' => 'Serbian',
    'sk' => 'Slovak',
    'sl' => 'Slovenian',
    'es' => 'Spanish',
    'sw' => 'Swahili',
    'sv' => 'Swedish',
    'ta' => 'Tamil',
    'te' => 'Telugu',
    'th' => 'Thai',
    'tr' => 'Turkish',
    'uk' => 'Ukrainian',
    'ur' => 'Urdu',
    'vi' => 'Vietnamese',
    'cy' => 'Welsh',
    'yi' => 'Yiddish'
];

// Load base list of english sentences
$sentences = file_get_contents('data/sentences.txt');
$sentences = explode("\r\n", $sentences);

// Will hold our intermediate format dataset
$dataSet = [];

// File containing finished data set
if(!file_exists('data/languagedataset.ser')) {
    // Setup initial dataset
    echo "Setting up initial dataset... ";
    foreach($languages as $languageCode => $languageName) {
        $dataSet[$languageCode] = [];
    }

    foreach($sentences as $sentence) {
        $dataSet['en'][sha1($sentence)] = $sentence;
    }
    file_put_contents('data/languagedataset.ser', serialize($dataSet));
    echo "Done" . PHP_EOL;
}
else {
    // Load existing dataset
    $dataSet = unserialize(file_get_contents('data/languagedataset.ser'));
}

// Check if any sentence has been added or removed from the list of english sentences
$datasetChecksums = array_keys($dataSet['en']);
$sentencesChecksums = [];
foreach($sentences as $sentence) {
    $sentenceChecksum = sha1($sentence);
    $sentencesChecksums[] = $sentenceChecksum;
    if(!in_array($sentenceChecksum, $datasetChecksums)) {
        echo "Adding new sentence id: " . $sentenceChecksum . " - " . $sentence . PHP_EOL;
        $dataSet['en'][$sentenceChecksum] = $sentence;
    }
}

// Now loop existing dataset and check if any sentences exist that have been removed from the sentences to use
foreach($datasetChecksums as $existingChecksum) {
    if(!in_array($existingChecksum, $sentencesChecksums)) {
        echo "Removing sentence id: " . $existingChecksum . " - " . $dataSet['en'][$existingChecksum] . PHP_EOL;

        // Remove sentence for each language
        foreach($languages as $languageCode => $languageName) {
            unset($dataSet[$languageCode][$existingChecksum]);
        }
    }
}

// Now sentences are up to date and we must look for untranslated sentences
// Keep sentences to translate
$sentencesToTranslate = [];
foreach($dataSet['en'] as $sentenceChecksum => $sentence) {
    foreach($languages as $languageCode => $languageName) {
        if($languageCode == 'en') continue; // Skip english
        // Check if translation is missing
        if(!array_key_exists($sentenceChecksum, $dataSet[$languageCode])) {
            $sentencesToTranslate[$languageCode][] = $sentenceChecksum;
            echo "Adding sentence for translation to " . $languageName . " id: " . $sentenceChecksum . " - " . $sentence . PHP_EOL;
        }
    }
}

if($sentencesToTranslate) {
    $gt = new GoogleTranslate('en', 'en', ['curl' => [CURLOPT_SSL_VERIFYPEER => false],  'verify' => __DIR__ . '/Data/cacert.pem']);

    // Loop each language and fetch translations for untranslated sentences
    foreach ($languages as $languageCode => $languageName) {
        if ($languageCode == 'en') continue; // Skip english

        // Since Google translate can block an IP for to many requests but can translate up to 5000 chars per request
        // we try to translate all the sentences in as few requests as possible
        $translatedSentences = [];
        $translationString = '';

        if(!array_key_exists($languageCode, $sentencesToTranslate)) {
            echo "No needed translations for " . $languageName . " skipping." . PHP_EOL;
            continue;
        }

        echo "Fetching translations for " . $languageName . PHP_EOL;

        foreach ($sentencesToTranslate[$languageCode] as $index => $sentenceChecksum) {
            // Add line of english sentence to translate
            $translationString .= $dataSet['en'][$sentenceChecksum] . "\r\n";

            if (strlen($translationString) > 4500 || end($sentencesToTranslate[$languageCode]) == $sentenceChecksum) {
                // Translate using Google Translate
                echo "Sending " . $languageName . " translation request to Google Translate... ";
                $translationString = $gt->setTarget($languageCode)->translate($translationString);
                echo "Done" . PHP_EOL;

                $translationResults = explode("\r\n", $translationString);
                $alreadyTranslated = count($translatedSentences);
                foreach ($translationResults as $resultIndex => $translationResult) {
                    $translatedSentences[$sentencesToTranslate[$languageCode][$resultIndex + $alreadyTranslated]] = $translationResult;
                }

                $translationString = '';
                $train = true;

                $sleep = rand(1, 5);
                echo "Sleeping for " . $sleep . "s" . PHP_EOL;
                sleep($sleep);
            }
        }

        foreach ($translatedSentences as $checksum => $translatedSentence) {
            echo "Storing " . $languageName . " translation for sentence id: " . $checksum . " - " . $translatedSentence . PHP_EOL;
            $dataSet[$languageCode][$checksum] = $translatedSentence;
        }
        file_put_contents('data/languagedataset.ser', serialize($dataSet));

        echo "Done" . PHP_EOL;
    }

    unset($sentencesToTranslate);
    unset($datasetChecksums);

    $sleep = rand(5, 10);
    echo "Sleeping for " . $sleep . "s" . PHP_EOL;
    sleep($sleep);
}
else {
    echo "Dataset up to date" . PHP_EOL;
}

// Model manager is used to store and retrieve model. Basically just serializing and storing the pipeline object
$modelManager = new ModelManager();

// Pipeline of TokenCountVectorizer, TfIdfTransformer and classifier.
// Without a pipeline one would manually have to store and restore the TokenCountVectorizer and TfIdfTransformer along
// with the Model in order to be able to make predictions in subsequent requests without retraining every time.
$pipeline = new Pipeline([
    new TokenCountVectorizer(new WordTokenizer()),
    new TfIdfTransformer(),
], new SVC(Kernel::RBF, 10000, 6));

try {
    if(file_exists('Data/model.dat')) {
        echo "Loading model... ";
        $pipeline = $modelManager->restoreFromFile('Data/model.dat');
        echo "Done" . PHP_EOL;
    }
    else {
        $train = true;
    }


    if($train) {

        echo "Model needs training!" . PHP_EOL;
        try {
            $samples = [];
            $targets = [];
            
            $cnt = 0;

            // To avoid running all languages for test we just use a few
            // We overwrite the old list so we don't have to change a lot of references to the array
            $languages = [
                'da' => 'Danish',
                'nl' => 'Dutch',
                'en' => 'English',
                'fi' => 'Finnish',
                'fr' => 'French',
                'de' => 'German',
                'it' => 'Italian',
                'no' => 'Norwegian',
                'pl' => 'Polish',
                'es' => 'Spanish',
                'sv' => 'Swedish'
            ];

            foreach ($languages as $languageCode => $languageName) {
                echo "Adding samples for " . $languageName;
                foreach ($dataSet[$languageCode] as $sample) {
                    $samples[] = $sample;
                    echo ".";
                    $targets[] = $languageCode;
                    if (++$cnt >= $numSentences) {
                        $cnt = 0;
                        echo PHP_EOL;
                        continue 2;
                    }
                }
                echo PHP_EOL;
            }

            unset($dataSet);

            echo "Creating ArrayDataset... ";
            $dataSet = new ArrayDataset($samples, $targets);
            echo "Done" . PHP_EOL;

            echo "Creating StratifiedRandomSplit... ";
            $randomSplit = new StratifiedRandomSplit($dataSet, 0.1);
            echo "Done" . PHP_EOL;

            echo "Training " . get_class($pipeline->getEstimator()) .  " classifier... ";
            $pipeline->train($randomSplit->getTrainSamples(), $randomSplit->getTrainLabels());
            echo "Done" . PHP_EOL;

            echo "Predicting labels... ";
            $predictedLabels = $pipeline->predict($randomSplit->getTestSamples());
            echo "Done" . PHP_EOL;

            echo 'Accuracy: ' . Accuracy::score($randomSplit->getTestLabels(), $predictedLabels) . PHP_EOL;

            echo "Storing model... ";
            $modelManager->saveToFile($pipeline, 'Data/model.dat');

            echo "Done" . PHP_EOL;
        } catch (\Exception $e) {
            echo "Error: " . $e->getMessage() . PHP_EOL;
        }
    }

    $msg = [];

    foreach ($argv as $index => $arg) {
        if($index == 0) continue;
        $msg[] = $arg;
    }

    $predictions = $pipeline->predict($msg);

    $result = [];

    foreach ($argv as $index => $arg) {
        if($index == 0) continue;
        $result[$arg] = $predictions[$index - 1];
    }

    //var_dump($result);
    echo json_encode($result);
}
catch(\Exception $e) {
    echo "Error: " . $e->getMessage() . PHP_EOL;
}
