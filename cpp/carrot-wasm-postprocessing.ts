import type {
    initialize as Iinitialize,
    CARROT_Postprocessing as ICARROT_Postprocessing,
    TreeringPostprocessingResult,
    CellsPostprocessingResult,
    CombinedPostprocessingResult,
    PairedPaths,
    PathPair,
    Path,
    Point,
    ImageSize,
} from "./carrot-wasm-postprocessing.d.ts"

import { 
    is_array_of_type,
    is_number_array,
} from "./dep.ts"



// for better readability
type fn_pointer = number;
type pointer = number;


type CARROT_Postprocessing_WASM = {
    _postprocess_treeringmapfile_wasm: (
        filesize:             number,
        read_file_callback_p: fn_pointer,
        read_file_handle:     number,
        workshape_width:      number,
        workshape_height:     number,
        og_shape_width:       number,
        og_shape_height:      number,
        // outputs
        treeringmap_workshape_png:       pointer,
        treeringmap_workshape_png_size:  pointer,
        ring_points_xy_json:             pointer,
        ring_points_xy_json_size:        pointer,
        returncode: pointer,
    ) => number,

    _postprocess_cellmapfile_wasm: (
        filesize:             number,
        read_file_callback_p: fn_pointer,
        read_file_handle:     number,
        workshape_width:      number,
        workshape_height:     number,
        og_shape_width:       number,
        og_shape_height:      number,
        // outputs
        cellmap_workshape_png:          pointer,
        cellmap_workshape_png_size:     pointer,
        instancemap_workshape_png:      pointer,
        instancemap_workshape_png_size: pointer,
        returncode: pointer,
    ) => number,

    _postprocess_combined_wasm: (
        cellmap_filesize:                      number,
        cellmap_read_file_callback_p:          fn_pointer,
        cellmap_read_file_callback_handle:     number,
        treeringmap_filesize:                  number,
        treeringmap_read_file_callback_p:      fn_pointer,
        treeringmap_read_file_callback_handle: number,
        workshape_width:      number,
        workshape_height:     number,
        og_shape_width:       number,
        og_shape_height:      number,
        // outputs
        cellmap_workshape_png:           pointer,
        cellmap_workshape_png_size:      pointer,
        instancemap_workshape_png:       pointer,
        instancemap_workshape_png_size:  pointer,
        treeringmap_workshape_png:       pointer,
        treeringmap_workshape_png_size:  pointer,
        ring_points_xy_json:             pointer,
        ring_points_xy_json_size:        pointer,
        ringmap_workshape_png_pp:        pointer,
        ringmap_workshape_png_size_p:    pointer,

        returncode: pointer,
    ) => number,

    _free_output: (p:pointer) => void,


    _malloc: (nbytes:number) => pointer,
    _free:   (ptr:pointer) => void,

    // deno-lint-ignore no-explicit-any
    addFunction: ( fn: (...args: any[]) => unknown, argtypes:string ) => number,
    
    HEAPU8: {
        set: (src:Uint8Array, dst:pointer) => void,
        slice: (start:number, end:number) => Uint8Array,
        [i:number]: number,
    }
    HEAP32: {
        [i:number]: number,
    },
    HEAP64: {
        [i:number]: bigint,
    },

    Asyncify: {
        // deno-lint-ignore no-explicit-any
        handleAsync: (fn:(...args:any[]) => Promise<unknown>) => unknown,
        currData: number|null,
    },
}


function wait(ms: number): Promise<unknown> {
    return new Promise((resolve) => {
        setTimeout(() => resolve(0), ms)
    })
}



export class CARROT_Postprocessing implements ICARROT_Postprocessing {
    constructor(private wasm:CARROT_Postprocessing_WASM){
        this.#read_file_callback_ptr = 
            wasm.addFunction(this.#read_file_callback, 'iiijj');
    }

    async postprocess_treeringmapfile(
        treeringmap: File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<TreeringPostprocessingResult | Error> {

        const handle:number = this.#handle_counter++;
        this.#read_file_callback_table[handle] = treeringmap;

        const treeringmap_workshape_png_pp:pointer     = this.#malloc(8);
        const treeringmap_workshape_png_size_p:pointer = this.#malloc(8);
        const ring_points_xy_json_pp:pointer           = this.#malloc(8);
        const ring_points_xy_json_size_p:pointer       = this.#malloc(8);
        const rc_ptr:pointer = this.#malloc(4, /*fill=*/255)

        let ring_points_xy_json_p:pointer|undefined = undefined;
        let treeringmap_workshape_png_p:pointer|undefined = undefined;

        try {

        let rc:number = await this.wasm._postprocess_treeringmapfile_wasm(
            treeringmap.size, 
            this.#read_file_callback_ptr, 
            handle, 
            work_size.width, 
            work_size.height,
            og_size.width,
            og_size.height,
            treeringmap_workshape_png_pp,
            treeringmap_workshape_png_size_p,
            ring_points_xy_json_pp,
            ring_points_xy_json_size_p,
            rc_ptr
        )
        // NOTE: the wasm function above returns before the execution is 
        // finished because of async issues, so currently polling until done
        while(this.wasm.Asyncify.currData != null)
            await wait(1);
        
        rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
        if(rc != 0)
            return new Error(`WASM error code = ${rc}`)

        ring_points_xy_json_p = this.wasm.HEAP32[ring_points_xy_json_pp >> 2]!;
        const ring_points_xy_json_size:number = 
            Number(this.wasm.HEAP64[ring_points_xy_json_size_p >> 3]);
        const ring_points_xy_json_u8:Uint8Array = this.wasm.HEAPU8.slice(
            ring_points_xy_json_p, 
            ring_points_xy_json_p + ring_points_xy_json_size
        )
        const obj:unknown = 
            JSON.parse(new TextDecoder().decode(ring_points_xy_json_u8));
        const paired_paths:PairedPaths|null = validate_paired_paths(obj)
        if(paired_paths == null)
            return new Error('WASM-JS communication inconcistencies')
        
        
        treeringmap_workshape_png_p = 
            this.wasm.HEAP32[treeringmap_workshape_png_pp >> 2]!;
        const treeringmap_workshape_png_size:number = 
            Number(this.wasm.HEAP64[treeringmap_workshape_png_size_p >> 3]);
        const treeringmap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
            treeringmap_workshape_png_p, 
            treeringmap_workshape_png_p + treeringmap_workshape_png_size
        )

        return {
            // @ts-ignore typescript is annoying
            treeringmap_workshape_png: new File([treeringmap_workshape_png_u8], 'treeringmap.png'),
            ring_points_xy: paired_paths,
        }
        
        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            this.wasm._free(treeringmap_workshape_png_pp);
            this.wasm._free(treeringmap_workshape_png_size_p);
            this.wasm._free(ring_points_xy_json_pp);
            this.wasm._free(ring_points_xy_json_size_p);
            this.wasm._free(rc_ptr);
            delete this.#read_file_callback_table[handle];

            if(ring_points_xy_json_p != undefined) 
                this.wasm._free_output(ring_points_xy_json_pp);
            
            console.warn('TODO: free output "treeringmap_workshape_png_p"!')
            //if(treeringmap_workshape_png_p != undefined)
            
        }
        
        return new Error('NOT IMPLEMENTED')
    }

    async postprocess_cellmapfile(
        cellmap:     File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CellsPostprocessingResult | Error> {

        const handle:number = this.#handle_counter++;
        this.#read_file_callback_table[handle] = cellmap;

        const cellmap_workshape_png_pp:pointer         = this.#malloc(8);
        const cellmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const instancemap_workshape_png_pp:pointer     = this.#malloc(8);
        const instancemap_workshape_png_size_p:pointer = this.#malloc(8);
        const rc_ptr:pointer = this.#malloc(4, /*fill=*/255)

        let cellmap_workshape_png_p:pointer|undefined = undefined;
        let instancemap_workshape_png_p:pointer|undefined = undefined;


        try {

        let rc:number = await this.wasm._postprocess_cellmapfile_wasm(
            cellmap.size, 
            this.#read_file_callback_ptr, 
            handle, 
            work_size.width, 
            work_size.height,
            og_size.width,
            og_size.height,
            cellmap_workshape_png_pp,
            cellmap_workshape_png_size_p,
            instancemap_workshape_png_pp,
            instancemap_workshape_png_size_p,
            rc_ptr
        )
        // NOTE: the wasm function above returns before the execution is 
        // finished because of async issues, so currently polling until done
        while(this.wasm.Asyncify.currData != null)
            await wait(1);

        rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
        if(rc != 0)
            return new Error(`WASM error code = ${rc}`)


        cellmap_workshape_png_p = 
            this.wasm.HEAP32[cellmap_workshape_png_pp >> 2]!;
        const cellmap_workshape_png_size:number = 
            Number(this.wasm.HEAP64[cellmap_workshape_png_size_p >> 3]);
        const cellmap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
            cellmap_workshape_png_p, 
            cellmap_workshape_png_p + cellmap_workshape_png_size
        )

        instancemap_workshape_png_p = 
            this.wasm.HEAP32[instancemap_workshape_png_pp >> 2]!;
        const instancemap_workshape_png_size:number = 
            Number(this.wasm.HEAP64[instancemap_workshape_png_size_p >> 3]);
        const instancemap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
            instancemap_workshape_png_p, 
            instancemap_workshape_png_p + instancemap_workshape_png_size
        )


        return {
            cellmap_workshape_png: 
                // @ts-ignore typescript is annoying
                new File([cellmap_workshape_png_u8], 'cellmap.png'),
            instancemap_workshape_png: 
                // @ts-ignore typescript is annoying
                new File([instancemap_workshape_png_u8], 'instancemap.png'),
        }

        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            this.wasm._free(cellmap_workshape_png_pp);
            this.wasm._free(cellmap_workshape_png_size_p);
            this.wasm._free(instancemap_workshape_png_pp);
            this.wasm._free(instancemap_workshape_png_size_p);
            this.wasm._free(rc_ptr);
            delete this.#read_file_callback_table[handle];

            if(cellmap_workshape_png_p != undefined) 
                this.wasm._free_output(cellmap_workshape_png_pp);
            if(instancemap_workshape_png_p != undefined) 
                this.wasm._free_output(instancemap_workshape_png_pp);
        }
    }

    async postprocess_combined(
        cellmap:     File,
        treeringmap: File,
        work_size:   ImageSize,
        og_size:     ImageSize,
    ): Promise<CombinedPostprocessingResult | Error> {
        const handle_cells:number = this.#handle_counter++;
        this.#read_file_callback_table[handle_cells] = cellmap;
        const handle_rings:number = this.#handle_counter++;
        this.#read_file_callback_table[handle_rings] = treeringmap;

        const rc_ptr:pointer = this.#malloc(4, /*fill=*/255)

        const cellmap_workshape_png_pp:pointer         = this.#malloc(8);
        const cellmap_workshape_png_size_p:pointer     = this.#malloc(8);
        const instancemap_workshape_png_pp:pointer     = this.#malloc(8);
        const instancemap_workshape_png_size_p:pointer = this.#malloc(8);
        const treeringmap_workshape_png_pp:pointer     = this.#malloc(8);
        const treeringmap_workshape_png_size_p:pointer = this.#malloc(8);
        const ring_points_xy_json_pp:pointer           = this.#malloc(8);
        const ring_points_xy_json_size_p:pointer       = this.#malloc(8);
        const ringmap_workshape_png_pp:pointer         = this.#malloc(8);
        const ringmap_workshape_png_size_p:pointer     = this.#malloc(8);

        let cellmap_workshape_png_p:pointer|undefined     = undefined;
        let instancemap_workshape_png_p:pointer|undefined = undefined;
        let treeringmap_workshape_png_p:pointer|undefined = undefined;
        let ring_points_xy_json_p:pointer|undefined       = undefined;
        let ringmap_workshape_png_p:pointer|undefined     = undefined;


        try {
            let rc:number = await this.wasm._postprocess_combined_wasm(
                cellmap.size, 
                this.#read_file_callback_ptr, 
                handle_cells, 
                treeringmap.size, 
                this.#read_file_callback_ptr, 
                handle_rings, 
                work_size.width, 
                work_size.height,
                og_size.width,
                og_size.height,

                cellmap_workshape_png_pp,
                cellmap_workshape_png_size_p,
                instancemap_workshape_png_pp,
                instancemap_workshape_png_size_p,
                treeringmap_workshape_png_pp,
                treeringmap_workshape_png_size_p,
                ring_points_xy_json_pp,
                ring_points_xy_json_size_p,
                ringmap_workshape_png_pp,
                ringmap_workshape_png_size_p,

                rc_ptr
            )
            // NOTE: the wasm function above returns before the execution is 
            // finished because of async issues, so currently polling until done
            while(this.wasm.Asyncify.currData != null)
                await wait(1);
    
            rc = (rc == 0)? this.wasm.HEAP32[rc_ptr >> 2]! : rc;
            if(rc != 0)
                return new Error(`WASM error code = ${rc}`)


            cellmap_workshape_png_p = 
                this.wasm.HEAP32[cellmap_workshape_png_pp >> 2]!;
            const cellmap_workshape_png_size:number = 
                Number(this.wasm.HEAP64[cellmap_workshape_png_size_p >> 3]);
            const cellmap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
                cellmap_workshape_png_p, 
                cellmap_workshape_png_p + cellmap_workshape_png_size
            )

            instancemap_workshape_png_p = 
                this.wasm.HEAP32[instancemap_workshape_png_pp >> 2]!;
            const instancemap_workshape_png_size:number = 
                Number(this.wasm.HEAP64[instancemap_workshape_png_size_p >> 3]);
            const instancemap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
                instancemap_workshape_png_p, 
                instancemap_workshape_png_p + instancemap_workshape_png_size
            )

            treeringmap_workshape_png_p = 
                this.wasm.HEAP32[treeringmap_workshape_png_pp >> 2]!;
            const treeringmap_workshape_png_size:number = 
                Number(this.wasm.HEAP64[treeringmap_workshape_png_size_p >> 3]);
            const treeringmap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
                treeringmap_workshape_png_p, 
                treeringmap_workshape_png_p + treeringmap_workshape_png_size
            )

            ring_points_xy_json_p = this.wasm.HEAP32[ring_points_xy_json_pp >> 2]!;
            const ring_points_xy_json_size:number = 
                Number(this.wasm.HEAP64[ring_points_xy_json_size_p >> 3]);
            const ring_points_xy_json_u8:Uint8Array = this.wasm.HEAPU8.slice(
                ring_points_xy_json_p, 
                ring_points_xy_json_p + ring_points_xy_json_size
            )
            const obj:unknown = 
                JSON.parse(new TextDecoder().decode(ring_points_xy_json_u8));
            const paired_paths:PairedPaths|null = validate_paired_paths(obj)
            if(paired_paths == null)
                return new Error('WASM-JS communication inconcistencies')

            ringmap_workshape_png_p = 
                this.wasm.HEAP32[ringmap_workshape_png_pp >> 2]!;
            const ringmap_workshape_png_size:number = 
                Number(this.wasm.HEAP64[ringmap_workshape_png_size_p >> 3]);
            const ringmap_workshape_png_u8:Uint8Array = this.wasm.HEAPU8.slice(
                ringmap_workshape_png_p, 
                ringmap_workshape_png_p + ringmap_workshape_png_size
            )

            return {
                cellmap_workshape_png: 
                    // @ts-ignore typescript is annoying
                    new File([cellmap_workshape_png_u8], 'cellmap.png'),
                instancemap_workshape_png: 
                    // @ts-ignore typescript is annoying
                    new File([instancemap_workshape_png_u8], 'instancemap.png'),
                
                treeringmap_workshape_png:
                    // @ts-ignore typescript is annoying
                    new File([treeringmap_workshape_png_u8], 'treeringmap.png'),
                ring_points_xy: paired_paths,

                ringmap_workshape_png:
                    // @ts-ignore typescript is annoying
                    new File([ringmap_workshape_png_u8], 'ringmap.png'),
            }
        } catch (e) {
            console.error('Unexpected error:', e)
            return e as Error;
        } finally {
            this.wasm._free(rc_ptr);
            this.wasm._free(cellmap_workshape_png_pp);
            this.wasm._free(cellmap_workshape_png_size_p);
            this.wasm._free(instancemap_workshape_png_pp);
            this.wasm._free(instancemap_workshape_png_size_p);
            this.wasm._free(treeringmap_workshape_png_pp);
            this.wasm._free(treeringmap_workshape_png_size_p);
            this.wasm._free(ring_points_xy_json_pp);
            this.wasm._free(ring_points_xy_json_size_p);
            this.wasm._free(ringmap_workshape_png_pp);
            this.wasm._free(ringmap_workshape_png_size_p);
            delete this.#read_file_callback_table[handle_cells];
            delete this.#read_file_callback_table[handle_rings];

            if(cellmap_workshape_png_p != undefined) 
                this.wasm._free_output(cellmap_workshape_png_pp);
            if(instancemap_workshape_png_p != undefined) 
                this.wasm._free_output(instancemap_workshape_png_pp);
            if(treeringmap_workshape_png_p != undefined) 
                this.wasm._free_output(treeringmap_workshape_png_pp);
            if(ringmap_workshape_png_p != undefined) 
                this.wasm._free_output(ringmap_workshape_png_pp);
            if(ring_points_xy_json_p != undefined) 
                this.wasm._free_output(ring_points_xy_json_pp);
        }
    }





    #handle_counter = 0;

    #read_file_callback_ptr:pointer;
    #read_file_callback_table: Record<number, File> = {};

    /** Called by WASM to read a required portion of a file */
    #read_file_callback = (
        handle: number,
        dstbuf: pointer,
        start:  bigint,
        size:   bigint,
    ): unknown  => {
        return this.wasm.Asyncify.handleAsync( async () => {
            const file:File|undefined = 
                this.#read_file_callback_table[handle];
            if(!file)
                return -1;
            
            const slice_u8:Uint8Array = new Uint8Array(
                await file.slice(Number(start), Number(start+size)).arrayBuffer()
            )
            this.wasm.HEAPU8.set(slice_u8, dstbuf);
            return 0;
        })
    }

    #malloc(nbytes:number, fill?:number): pointer {
        const p:pointer = this.wasm._malloc(nbytes);
        this.wasm.HEAPU8.set(new Uint8Array(nbytes).fill(fill ?? 0), p)
        return p;
    }
}


function validate_point(x:unknown): Point|null {
    if(is_number_array(x) && x.length == 2){
        return x as Point;
    }
    else return null;
}

function validate_path(x:unknown): Path|null {
    if(is_array_of_type(x, validate_point)){
        return x;
    }
    else return null;
}

function validate_pathpair(x:unknown): PathPair|null {
    if(is_array_of_type(x, validate_path) && x.length == 2){
        return x as PathPair;
    }
    else return null;
}

function validate_paired_paths(x:unknown):PairedPaths|null {
    if(is_array_of_type(x, validate_pathpair)){
        return x;
    }
    else return null;
}




export const initialize:typeof Iinitialize = async () => {
    const wasm:CARROT_Postprocessing_WASM = await (
        await import('./build-wasm/carrot_postprocessing_wasm.js')
    // deno-lint-ignore no-explicit-any
    ).default() as any;

    return new CARROT_Postprocessing(wasm);
}


