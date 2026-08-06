<?php
/**
 * Smoke test for LlamaEmbed.
 * Usage: php -d extension=…/modules/llama.so test_llama.php [gguf-path]
 */
if (!extension_loaded('llama')) {
    fwrite(STDERR, "SKIP: llama extension not loaded\n");
    exit(0);
}

$path = !empty($argv[1])
    ? $argv[1]
    : '/home/press/Llama/models/bge-m3/bge-m3-q4_k_m.gguf';

if (!is_readable($path)) {
    fwrite(STDERR, "FAIL: GGUF not readable: {$path}\n");
    exit(1);
}

$e = LlamaEmbed::load($path, array(
    'n_gpu_layers' => 99,
    'pooling' => 'cls',
));
assert($e->dimension() === 1024, 'dimension must be 1024 for bge-m3');

$vec = $e->embed('Media Outreach press release translation test.');
assert(is_array($vec) && count($vec) === 1024, 'embed length');
$sum = 0.0;
foreach ($vec as $x) {
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

$e->close();

$e2 = LlamaEmbed::load($path, array('n_gpu_layers' => 99, 'pooling' => 'cls'));
$vec2 = $e2->embed('second load');
assert(count($vec2) === 1024, 'second load embed');
$e2->close();

fwrite(STDERR, "test_llama.php OK\n");
