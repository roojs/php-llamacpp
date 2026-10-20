<?php
/*
   +----------------------------------------------------------------------+
   | Copyright (c) 2026 Roo J Solutions Ltd.                              |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | https://www.php.net/license/3_01.txt                                 |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
   | Author: Leon <leon@roojs.com>                                        |
   +----------------------------------------------------------------------+
*/

/**
 * Smoke test for LlamaEmbed.
 * Usage: php -d extension=…/modules/llama.so test_llama.php <gguf-path>
 */
if (!extension_loaded('llama')) {
    fwrite(STDERR, "SKIP: llama extension not loaded\n");
    exit(0);
}

if (empty($argv[1])) {
    fwrite(STDERR, "Usage: php test_llama.php <gguf-path>\n");
    exit(1);
}
$path = $argv[1];

if (!is_readable($path)) {
    fwrite(STDERR, "FAIL: GGUF not readable: {$path}\n");
    exit(1);
}

$e = LlamaEmbed::load($path);
assert($e->dimension() === 1024, 'dimension must be 1024 for bge-m3');

$vec = $e->embedBatch(array('The quick brown fox jumps over the lazy dog.'));
assert(is_array($vec) && count($vec) === 1 && count($vec[0]) === 1024, 'embed length');
$sum = 0.0;
foreach ($vec[0] as $x) {
    $sum += $x * $x;
}
assert($sum > 0.9 && $sum < 1.1, 'L2-normalized roughly unit length');

$batch = $e->embedBatch(array(
    'hello', 'world', 'alpha', 'beta', 'gamma',
    'delta', 'epsilon', 'zeta', 'eta', 'theta',
));
assert(count($batch) === 10, 'embedBatch count 10');
assert(count($batch[0]) === 1024 && count($batch[9]) === 1024, 'embedBatch dims');
/* packed batch should still differ by text */
assert($batch[0] !== $batch[1], 'different texts → different vectors');

unset($e);

$e2 = LlamaEmbed::load($path);
$vec2 = $e2->embedBatch(array('second load'));
assert(count($vec2) === 1 && count($vec2[0]) === 1024, 'second load embed');
unset($e2);

fwrite(STDERR, "test_llama.php OK\n");
