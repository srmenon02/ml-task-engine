export default function LoadingSpinner({ size = 'medium', text = 'Loading...'}){
    const sizes = {
        mall: 'w-4 h-4 border-2',
        medium: 'w-8 h-8 border-3',
        large: 'w-12 h-12 border-4',
    }

    return (
        <div className="flex flex-col items-center justify-center gap-3 py-8">
            <div
                className={`${sizes[size]} border-blue-600 border-t-transparent rounded-full animate-spin`}
            />
            <p className="text=gray=500 text-sm">{text}</p>
        </div>
    );
}