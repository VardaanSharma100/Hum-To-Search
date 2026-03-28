import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import gsap from 'gsap';

const AudioVisualizer = ({ status, audioData }) => {
    const meshRef = useRef();

    // 1000 particles
    const count = 1000;

    // Particle data
    const particles = useMemo(() => {
        const temp = [];
        const colorGen = new THREE.Color();
        for (let i = 0; i < count; i++) {
            // Random position for sphere
            const phi = Math.acos(-1 + (2 * i) / count);
            const theta = Math.sqrt(count * Math.PI) * phi;
            const spherePos = new THREE.Vector3(
                Math.cos(theta) * Math.sin(phi) * 5,
                Math.sin(theta) * Math.sin(phi) * 5,
                Math.cos(phi) * 5
            );

            // Horizontal line position
            const lineX = (i / count) * 40 - 20;
            const linePos = new THREE.Vector3(lineX, 0, 0);

            // Tight sphere position
            const tightPos = spherePos.clone().multiplyScalar(0.2);

            // Colors: vibrant blues, reds, oranges, yellows
            const colors = ['#0077ff', '#ff3300', '#ff9900', '#ffcc00', '#00e5ff'];
            colorGen.set(colors[i % colors.length]);

            temp.push({
                currentPos: spherePos.clone(),
                spherePos,
                linePos,
                tightPos,
                color: colorGen.clone(),
                id: i
            });
        }
        return temp;
    }, [count]);

    const dummy = useMemo(() => new THREE.Object3D(), []);

    // Set colors once on mount
    useEffect(() => {
        if (!meshRef.current) return;
        const colorArray = new Float32Array(count * 3);
        particles.forEach((p, i) => {
            p.color.toArray(colorArray, i * 3);
        });
        meshRef.current.geometry.setAttribute('color', new THREE.InstancedBufferAttribute(colorArray, 3));
    }, [particles]);

    // Handle state transitions with GSAP
    useEffect(() => {
        if (!meshRef.current) return;

        let targetKey = 'linePos';
        if (status === 'recording') targetKey = 'spherePos';
        if (status === 'loading') targetKey = 'tightPos';
        if (status === 'success') targetKey = 'linePos'; // or scatter

        particles.forEach((p) => {
            gsap.to(p.currentPos, {
                x: p[targetKey].x,
                y: p[targetKey].y,
                z: p[targetKey].z,
                duration: 2.0,
                ease: 'power3.inOut'
            });
        });
    }, [status, particles]);

    useFrame((state) => {
        if (!meshRef.current) return;

        // Rotation animation
        if (status === 'recording') {
            meshRef.current.rotation.y += 0.003;
            meshRef.current.rotation.x += 0.002;
        } else if (status === 'loading') {
            meshRef.current.rotation.y += 0.04;
            meshRef.current.rotation.x += 0.02;
        } else {
            meshRef.current.rotation.x = THREE.MathUtils.lerp(meshRef.current.rotation.x, 0, 0.05);
            meshRef.current.rotation.y = THREE.MathUtils.lerp(meshRef.current.rotation.y, 0, 0.05);
            meshRef.current.rotation.z = THREE.MathUtils.lerp(meshRef.current.rotation.z, 0, 0.05);
        }

        const t = state.clock.getElapsedTime();

        // Update positions
        particles.forEach((p, i) => {
            let px = p.currentPos.x;
            let py = p.currentPos.y;
            let pz = p.currentPos.z;

            // Floating sinusoidal wave in idle
            if (status === 'idle' || status === 'success') {
                py += Math.sin(t * 2 + px * 0.5) * 0.5;
            }

            // Audio Reactivity during recording on the sphere
            if (status === 'recording' && audioData && audioData.length > 0) {
                // Map particle index to frequency bin
                const bin = Math.floor((i / count) * (audioData.length * 0.5));
                const val = audioData[bin] || 0;

                let intensity = (val / 255.0);

                // Clump and jagged peaks effect outward from the origin
                if (intensity > 0.15) {
                    const dist = Math.sqrt(px * px + py * py + pz * pz) || 1;
                    const push = intensity * 4.0 * (i % 2 === 0 ? 1 : -0.2);
                    px += (px / dist) * push;
                    py += (py / dist) * push;
                    pz += (pz / dist) * push;
                }
            }

            dummy.position.set(px, py, pz);
            dummy.updateMatrix();
            meshRef.current.setMatrixAt(i, dummy.matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[null, null, count]}>
            <sphereGeometry args={[0.08, 16, 16]} />
            <meshPhysicalMaterial vertexColors={true} emissive={"#222"} roughness={0.2} metalness={0.8} />
        </instancedMesh>
    );
};

export default AudioVisualizer;