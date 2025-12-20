import { initialize } from "../cpp/carrot-wasm-postprocessing.ts"

import { asserts } from "./dep.ts"





Deno.test('postprocess_treeringmapfile', async () => {
    const module = await initialize()

    const invalidfile = new File([], 'invalid.png')
    const worksize = {width: 555, height:555}
    const og_size = {width:2048, height:2048}

    const output0 = await module.postprocess_combined(null, invalidfile, worksize, og_size)
    asserts.assertInstanceOf(output0, Error)
    // make sure no c++ error
    asserts.assert(
        !output0.message.toLowerCase().includes('aborted'), 
        'Not a clean C++ exit'
    )



    const filepath0 = import.meta.resolve('./assets/treeringsmap0.png').replace('file://','')
    const treeringmapfile0 = new File([Deno.readFileSync(filepath0)], 'treerrings.png')

    const output1 = await module.postprocess_combined(null, treeringmapfile0, worksize, og_size)
    //console.log(output1)
    asserts.assertNotInstanceOf(output1, Error)
    asserts.assert('ring_points_xy' in output1)
    asserts.assertEquals(output1.ring_points_xy.length, 5)



    const filepath1 = import.meta.resolve('./assets/treeringsmap1.png').replace('file://','')
    const treeringmapfile1 = new File([Deno.readFileSync(filepath1)], 'treerrings.png')

    const worksize2 = {width: 3297, height:4379}
    const og_size2 = {width:3297, height:4379}
    const output2 = await module.postprocess_combined(null, treeringmapfile1, worksize2, og_size2)
    asserts.assertNotInstanceOf(output2, Error)
    asserts.assert('ring_points_xy' in output2)
    asserts.assertEquals(output2.ring_points_xy.length, 3)
})


Deno.test('postprocess_cellmapfile', async () => {
    const module = await initialize()

    const invalidfile = new File([], 'invalid.png')
    const worksize = {width: 555, height:555}
    const og_size = {width:2048, height:2048}

    const output0 = await module.postprocess_combined(invalidfile, null, worksize, og_size)
    asserts.assertInstanceOf(output0, Error)
    // make sure no c++ error
    asserts.assert(
        !output0.message.toLowerCase().includes('aborted'), 
        'Not a clean C++ exit'
    )


    const filepath0 = import.meta.resolve('./assets/cellmap0.png').replace('file://','')
    const cellmapfile0 = new File([Deno.readFileSync(filepath0)], 'cellmap.png')

    const output1 = await module.postprocess_combined(cellmapfile0, null, worksize, og_size)
    //console.log(output1)
    asserts.assertNotInstanceOf(output1, Error)
    asserts.assert('instancemap_workshape_png' in output1)
})


Deno.test('postprocess_combined', async () => {
    const module = await initialize()

    const invalidfile = new File([], 'invalid.png')
    const worksize = {width: 555, height:555}
    const og_size = {width:2048, height:2048}

    const output0 = await module.postprocess_combined(invalidfile, invalidfile, worksize, og_size)
    asserts.assertInstanceOf(output0, Error)
    // make sure no c++ error
    asserts.assert(
        !output0.message.toLowerCase().includes('aborted'), 
        'Not a clean C++ exit'
    )


    const filepath0 = import.meta.resolve('./assets/cellmap3-combined.png').replace('file://','')
    const cellmapfile0 = new File([Deno.readFileSync(filepath0)], 'cellmap3-combined.png')
    const filepath1 = import.meta.resolve('./assets/treeringsmap3-combined.png').replace('file://','')
    const treeringmapfile1 = new File([Deno.readFileSync(filepath1)], 'treeringsmap3-combined.png')

    const output1 = await module.postprocess_combined(cellmapfile0, treeringmapfile1, worksize, og_size)
    //console.log(output1)
    asserts.assertNotInstanceOf(output1, Error)
    asserts.assert('ringmap_workshape_png' in output1)

    asserts.assertGreater(output1.cell_info.length, 0)
})



Deno.test('resize_mask', async () => {
    const module = await initialize();
    
    const filepath1 = import.meta.resolve('./assets/treeringsmap1.png').replace('file://','')
    const treeringmapfile1 = new File([Deno.readFileSync(filepath1)], 'treerrings.png')


    const worksize = {width:400, height:400}
    const targetsize = {width:10001, height:10002}
    const encoded = await module.resize_mask(treeringmapfile1, worksize, targetsize)
    asserts.assertInstanceOf(encoded, File)
})
